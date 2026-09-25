// C++ headers
#ifdef DUMP_GPU_TK_TUPLES
#include <mutex>
#endif

// Alpaka headers
#include <alpaka/alpaka.hpp>

// CMSSW headers
// #include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

// local headers
#include "TrackSoAMergerKernels.h"
#include "TrackSoAMergerKernelsImpl.h"

// #define GPU_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  TrackSoAMergerKernels::TrackSoAMergerKernels(Params const &params, Queue &queue) : params_(params) {
    counters_d_ = reco::TrackMergerCounterSoACollection(queue, params_.maxTracks);
    totCounters_ = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, 2u);

    totTracks_ = cms::alpakatools::make_device_view(queue, *reinterpret_cast<uint32_t *>(totCounters_->data()));
    totHits_ = cms::alpakatools::make_device_view(queue, *reinterpret_cast<uint32_t *>(totCounters_->data() + 1));

    alpaka::memset(queue, *totTracks_, 0);
    alpaka::memset(queue, *totHits_, 0);
  }

  reco::TracksSoACollection TrackSoAMergerKernels::makeMergedTracks(Queue &queue,
                                                                    ::mergerKernels::InputTracks const &allTracks) {
    countGoodTracks(queue, allTracks);
    fillGoodTracks(queue, allTracks);
    filterTracks(queue);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "finished filtering track SoAs on GPU" << std::endl;
#endif

    return std::move(*tracks_d_);
  }

  void TrackSoAMergerKernels::countGoodTracks(Queue &queue, ::mergerKernels::InputTracks const &allTracks) {
    using namespace trackSoAMergerKernels;

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Starting TrackSoAMergerKernels::countGoodTracks" << std::endl;
#endif

    if (allTracks.nTracks != 0) {
      const auto threadsPerBlock = 128u;
      const auto blocks = cms::alpakatools::divide_up_by(allTracks.nTracks, threadsPerBlock);
      const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_countGoodTracks{},
                          allTracks,
                          params_.minQuality,
                          counters_d_->view(),
                          totTracks_->data(),
                          totHits_->data());
    }

    auto totCountersHost = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, 2);
    alpaka::memcpy(queue, totCountersHost, *totCounters_);
    alpaka::wait(queue);

#ifdef GPU_DEBUG
    std::cout << "Kernel_countGoodTracks -> done!" << std::endl;
    std::cout << "Total good tracks: " << totCountersHost[0] << ", total hits: " << totCountersHost[1] << std::endl;
#endif

    tracks_d_ = reco::TracksSoACollection(queue, totCountersHost[0], totCountersHost[1]);
    duplicate_d_ = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, totCountersHost[0]);
    alpaka::memset(queue, *duplicate_d_, 0);

    if (totCountersHost[0] > 0) {
      constexpr auto threadsPrefixScan = 256u;
      auto blocksPrefixScan = (totCountersHost[0] + threadsPrefixScan - 1) / threadsPrefixScan;
      auto workDivPrefixScan = cms::alpakatools::make_workdiv<Acc1D>(blocksPrefixScan, threadsPrefixScan);
      auto bCounter = cms::alpakatools::make_device_buffer<int32_t>(queue);
      alpaka::memset(queue, bCounter, 0);

      // Launch to build the hit offsets for the selected tracks
      // tracks().hitOffsets() holds the ending offset of each track hits vector
      // so here we simply do the prefix sum of the number of hits per track
      alpaka::exec<Acc1D>(queue,
                          workDivPrefixScan,
                          cms::alpakatools::multiBlockPrefixScan<uint32_t>(),
                          counters_d_->view().hitsInTrack().data(),
                          tracks_d_->view().tracks().hitOffsets().data(),
                          totCountersHost[0],
                          blocksPrefixScan,
                          bCounter.data(),
                          alpaka::getPreferredWarpSize(alpaka::getDev(queue)));
    }
  }

  void TrackSoAMergerKernels::fillGoodTracks(Queue &queue, ::mergerKernels::InputTracks const &allTracks) {
    using namespace trackSoAMergerKernels;

    if (tracks_d_->view().tracks().metadata().size() > 0) {
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Starting TrackSoAMergerKernels::fillGoodTracks" << std::endl;
#endif
      const auto threadsPerBlock = 128u;
      const auto blocks = cms::alpakatools::divide_up_by(tracks_d_->view().tracks().metadata().size(), threadsPerBlock);
      const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);

      alpaka::exec<Acc1D>(queue,
                          workDiv1D,
                          Kernel_fillGoodTracks{},
                          allTracks,
                          counters_d_->view(),
                          tracks_d_->view().tracks(),
                          tracks_d_->view().trackHits());
    }
#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Kernel_fillGoodTracks -> done!" << std::endl;
#endif
  }

  void TrackSoAMergerKernels::filterTracks(Queue &queue) {
    using namespace trackSoAMergerKernels;

#ifdef GPU_DEBUG
    alpaka::wait(queue);
    std::cout << "Starting TrackSoAMergerKernels::filterTracks" << std::endl;
#endif

    auto const nTracks = tracks_d_->view().tracks().metadata().size();
    if (nTracks < 2) {
      return;
    }

    if (params_.doSameHitsDuplicates or params_.doParamDuplicates) {
      constexpr uint32_t tracksPerBlock = 8;
      constexpr uint32_t comparisonsPerBlock = 32;

      auto const blocksX = cms::alpakatools::divide_up_by(nTracks, tracksPerBlock);

      auto const blocksY = std::min(cms::alpakatools::divide_up_by(nTracks, comparisonsPerBlock), 65535u);

      Vec2D const blocks{blocksX, blocksY};
      Vec2D const threads{tracksPerBlock, comparisonsPerBlock};

      auto const workDiv2D = cms::alpakatools::make_workdiv<Acc2D>(blocks, threads);

      // #ifdef GPU_DEBUG
      //     alpaka::wait(queue);
      //     printf("filterTracks: nTracks %d, blocksX %d, blocksY %d tracksPerBlock %d\n", nTracks, blocksX, blocksY, tracksPerBlock);
      // #endif
      //     if (params_.doSameHitsDuplicates) {
      //       alpaka::exec<Acc2D>(queue,
      //                           workDiv2D,
      //                           Kernel_sameHitsDuplicates{},
      //                           tracks_d_->view().tracks(),
      //                           tracks_d_->view().trackHits(),
      //                           params_.matchFraction,
      //                           params_.dupMinHits);

      // #ifdef GPU_DEBUG
      //     alpaka::wait(queue);
      //     std::cout << "Kernel_sameHitsDuplicates -> done!" << std::endl;
      // #endif
      //       }

      //     if(params_.doParamDuplicates) {
      //         alpaka::exec<Acc2D>(queue,
      //                             workDiv2D,
      //                             Kernel_trackParameterDuplicates{},
      //                             tracks_d_->view().tracks(),
      //                             params_.dupNSigma2,
      //                             params_.dupMaxDeltaR2,
      //                             params_.dupPtDifference);
      // #ifdef GPU_DEBUG
      //     alpaka::wait(queue);
      //     std::cout << "Kernel_trackParameterDuplicates -> done!" << std::endl;
      // #endif
      //         }
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      printf("filterTracks: nTracks %d, blocksX %d, tracksPerBlock %d\n", nTracks, blocksX, tracksPerBlock);
#endif

      //   auto const workDiv2D =
      //       cms::alpakatools::make_workdiv<Acc2D>(blocks, threads);

      alpaka::exec<Acc2D>(queue,
                          workDiv2D,
                          Kernel_trackDuplicates{},
                          tracks_d_->view().tracks(),
                          tracks_d_->view().trackHits(),
                          params_.doSameHitsDuplicates,
                          params_.doParamDuplicates,
                          params_.matchFraction,
                          params_.dupMinHits,
                          params_.dupNSigma2,
                          params_.dupMaxDeltaR2,
                          params_.dupPtDifference,
                          duplicate_d_->data());

      constexpr uint32_t threadsPerBlock = 128;
      auto const blocks1D = cms::alpakatools::divide_up_by(nTracks, threadsPerBlock);
      auto const workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks1D, threadsPerBlock);

      alpaka::exec<Acc1D>(
          queue, workDiv, Kernel_applyTrackDuplicates{}, tracks_d_->view().tracks(), duplicate_d_->data());
#ifdef GPU_DEBUG
      alpaka::wait(queue);
      std::cout << "Kernel_trackDuplicates + Kernel_applyTrackDuplicates -> done!" << std::endl;
#endif
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
