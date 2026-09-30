#ifndef RecoTracker_FinalTrackSelectors_plugins_alpaka_TrackSoAMergerKernels_h
#define RecoTracker_FinalTrackSelectors_plugins_alpaka_TrackSoAMergerKernels_h

// #define GPU_DEBUG
// #define DUMP_GPU_TK_TUPLES

#include <cstdint>
#include <utility>

#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TrackDefinitions.h"
#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "DataFormats/SoATemplate/interface/SoAConstMultiView.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AtomicPairCounter.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HistoContainer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/FinalTrackSelectors/interface/TrackMergerCounterSoA.h"
#include "RecoTracker/FinalTrackSelectors/interface/alpaka/TrackMergerCounterSoACollection.h"

namespace mergerKernels {
  constexpr uint32_t maxTrackSoACollections = 20;

  using TracksConstView = reco::TrackSoAConstView;
  using TrackHitsConstView = reco::TrackHitSoAConstView;
  using TracksMultiView = SoAConstMultiView<TracksConstView, maxTrackSoACollections>;
  using TrackHitsMultiView = SoAConstMultiView<TrackHitsConstView, maxTrackSoACollections>;

  struct InputTracks {
    reco::TrackSoAConstView views[maxTrackSoACollections];
    reco::TrackHitSoAConstView hitViews[maxTrackSoACollections];
    int nInputs;
    int nTracks;
  };

  struct Params {
    bool doSameHitsDuplicates;
    bool doParamDuplicates;
    pixelTrack::Quality minQuality;

    int maxTracks;

    // duplicate filtering criteria
    int dupMinHits;
    double matchFraction;
    double dupNSigma2;
    double dupMaxDeltaR2;
    double dupPtDifference;
  };
}  // namespace mergerKernels

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace ::mergerKernels;

  class TrackSoAMergerKernels {
  public:
    TrackSoAMergerKernels() = default;
    ~TrackSoAMergerKernels() = default;

    TrackSoAMergerKernels(const TrackSoAMergerKernels&) = delete;
    TrackSoAMergerKernels(TrackSoAMergerKernels&&) = delete;
    TrackSoAMergerKernels& operator=(const TrackSoAMergerKernels&) = delete;
    TrackSoAMergerKernels& operator=(TrackSoAMergerKernels&&) = delete;

    TrackSoAMergerKernels(Queue& queue, Params const& params);

    reco::TracksSoACollection makeMergedTracks(Queue& queue,
                                               const TracksMultiView& tracks,
                                               const TrackHitsMultiView& hits);

    void countGoodTracks(Queue& queue, const TracksMultiView& tracks);
    void fillGoodTracks(Queue& queue, const TracksMultiView& tracks, const TrackHitsMultiView& hits);

    void filterTracks(Queue& queue);

  private:
    Params const& params_;

    std::optional<reco::TracksSoACollection> tracks_d_;
    std::optional<reco::TrackMergerCounterSoACollection> counters_d_;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> totCounters_;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> duplicate_d_;
    std::optional<cms::alpakatools::device_view<Device, uint32_t>> totTracks_;
    std::optional<cms::alpakatools::device_view<Device, uint32_t>> totHits_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_FinalTrackSelectors_plugins_alpaka_TrackSoAMergerKernels_h
