#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"

#include "LSTEvent.h"

#include "Hit.h"
#include "Kernels.h"
#include "MiniDoublet.h"
#include "PixelQuintuplet.h"
#include "PixelTriplet.h"
#include "Quintuplet.h"
#include "Segment.h"
#include "TrackCandidate.h"
#include "Triplet.h"
#include "Quadruplet.h"

using Device = ALPAKA_ACCELERATOR_NAMESPACE::Device;
using Queue = ALPAKA_ACCELERATOR_NAMESPACE::Queue;
using Acc1D = ALPAKA_ACCELERATOR_NAMESPACE::Acc1D;
using Acc3D = ALPAKA_ACCELERATOR_NAMESPACE::Acc3D;

using namespace ALPAKA_ACCELERATOR_NAMESPACE::lst;

void LSTEvent::initSync() {
  alpaka::wait(queue_);  // other calls can be asynchronous

  //reset the arrays
  for (int i = 0; i < 6; i++) {
    n_minidoublets_by_layer_barrel_[i] = 0;
    n_segments_by_layer_barrel_[i] = 0;
    n_triplets_by_layer_barrel_[i] = 0;
    n_quintuplets_by_layer_barrel_[i] = 0;
    n_quadruplets_by_layer_barrel_[i] = 0;
    if (i < 5) {
      n_minidoublets_by_layer_endcap_[i] = 0;
      n_segments_by_layer_endcap_[i] = 0;
      n_triplets_by_layer_endcap_[i] = 0;
      n_quintuplets_by_layer_endcap_[i] = 0;
      n_quadruplets_by_layer_endcap_[i] = 0;
    }
  }
}

void LSTEvent::resetEventSync() {
  alpaka::wait(queue_);  // synchronize to reset consistently
  //reset the arrays
  for (int i = 0; i < 6; i++) {
    n_minidoublets_by_layer_barrel_[i] = 0;
    n_segments_by_layer_barrel_[i] = 0;
    n_triplets_by_layer_barrel_[i] = 0;
    n_quintuplets_by_layer_barrel_[i] = 0;
    n_quadruplets_by_layer_barrel_[i] = 0;
    if (i < 5) {
      n_minidoublets_by_layer_endcap_[i] = 0;
      n_segments_by_layer_endcap_[i] = 0;
      n_triplets_by_layer_endcap_[i] = 0;
      n_quintuplets_by_layer_endcap_[i] = 0;
      n_quadruplets_by_layer_endcap_[i] = 0;
    }
  }
  lstInputDC_ = nullptr;
  hitsDC_.reset();
  rangesDC_.reset();
  miniDoubletsDC_.reset();
  segmentsDC_.reset();
  pixelSegmentsDC_.reset();
  tripletsDC_.reset();
  quintupletsDC_.reset();
  trackCandidatesBaseDC_.reset();
  trackCandidatesExtendedDC_.reset();
  pixelTripletsDC_.reset();
  pixelQuintupletsDC_.reset();
  quadrupletsDC_.reset();

  lstInputHC_.reset();
  hitsHC_.reset();
  rangesHC_.reset();
  miniDoubletsHC_.reset();
  segmentsHC_.reset();
  pixelSegmentsHC_.reset();
  tripletsHC_.reset();
  quintupletsHC_.reset();
  pixelTripletsHC_.reset();
  pixelQuintupletsHC_.reset();
  trackCandidatesBaseHC_.reset();
  trackCandidatesExtendedHC_.reset();
  modulesHC_.reset();
  quadrupletsHC_.reset();
}

void LSTEvent::addInputToEvent(LSTInputDeviceCollection const* lstInputDC) {
  lstInputDC_ = lstInputDC;

  pixelSize_ = lstInputDC_->size()[1];
  pixelModuleIndex_ = pixelMapping_.pixelModuleIndex;
}

void LSTEvent::addHitToEvent() {
  if (!hitsDC_) {
    const int32_t nHits = lstInputDC_->size()[0];
    hitsDC_.emplace(queue_, nHits, static_cast<int32_t>(nModules_));
    auto buf = hitsDC_->buffer();
    alpaka::memset(queue_, buf, 0xff);
  }

  if (!rangesDC_) {
    rangesDC_.emplace(nLowerModules_ + 1, queue_);
    auto buf = rangesDC_->buffer();
    alpaka::memset(queue_, buf, 0xff);
  }

  auto const hit_loop_workdiv = cms::alpakatools::make_workdiv<Acc1D>(max_blocks, 256);

  alpaka::exec<Acc1D>(queue_,
                      hit_loop_workdiv,
                      HitLoopKernel{},
                      Endcap,
                      TwoS,
                      nModules_,
                      nEndCapMap_,
                      endcapGeometry_.const_view(),
                      modules_.const_view().modules(),
                      lstInputDC_->const_view().hits(),
                      hitsDC_->view().extended(),
                      hitsDC_->view().ranges());

  auto const module_ranges_workdiv = cms::alpakatools::make_workdiv<Acc1D>(max_blocks, 256);

  alpaka::exec<Acc1D>(queue_,
                      module_ranges_workdiv,
                      ModuleRangesKernel{},
                      modules_.const_view().modules(),
                      hitsDC_->view().ranges(),
                      nLowerModules_);
}

void LSTEvent::addPixelSegmentToEventStart() {
  if (pixelSize_ == n_max_pixel_segments_per_module) {
    lstWarning(
        "\
          *********************************************************\n\
          * Warning: Pixel line segments may be truncated.        *\n\
          * You need to increase n_max_pixel_segments_per_module. *\n\
          *********************************************************");
  }

  if (!pixelSegmentsDC_) {
    pixelSegmentsDC_.emplace(pixelSize_, queue_);
  }
}

void LSTEvent::addPixelSegmentToEventFinalize() {
  auto const addPixelSegmentToEvent_workdiv = cms::alpakatools::make_workdiv<Acc1D>(max_blocks, 256);

  alpaka::exec<Acc1D>(queue_,
                      addPixelSegmentToEvent_workdiv,
                      AddPixelSegmentToEventKernel{},
                      modules_.const_view().modules(),
                      rangesDC_->const_view(),
                      lstInputDC_->const_view().hits(),
                      hitsDC_->view().extended(),
                      lstInputDC_->const_view().pixelSeeds(),
                      miniDoubletsDC_->view().miniDoublets(),
                      segmentsDC_->view().segments(),
                      pixelSegmentsDC_->view(),
                      pixelModuleIndex_,
                      pixelSize_);
}

void LSTEvent::createMiniDoublets() {
  if (!miniDoubletsDC_) {
    // Create a view for the element nLowerModules_ inside rangesOccupancy->miniDoubletModuleOccupancy
    auto rangesOccupancy = rangesDC_->view();
    auto dst_view_miniDoubletModuleOccupancy =
        cms::alpakatools::make_device_view(queue_, rangesOccupancy.miniDoubletModuleOccupancy()[nLowerModules_]);

    // Create a host buffer for a value to be passed to the device
    auto pixelMaxMDs_buf_h = cms::alpakatools::make_host_buffer<int>(queue_);
    *pixelMaxMDs_buf_h.data() = n_max_pixel_md_per_modules;

    alpaka::memcpy(queue_, dst_view_miniDoubletModuleOccupancy, pixelMaxMDs_buf_h);

    auto dst_view_miniDoubletModuleOccupancyPix =
        cms::alpakatools::make_device_view(queue_, rangesOccupancy.miniDoubletModuleOccupancy()[pixelModuleIndex_]);

    alpaka::memcpy(queue_, dst_view_miniDoubletModuleOccupancyPix, pixelMaxMDs_buf_h);

    auto const createMDArrayRangesGPU_workDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 1024);

    alpaka::exec<Acc1D>(queue_,
                        createMDArrayRangesGPU_workDiv,
                        CreateMDArrayRangesGPU{},
                        modules_.const_view().modules(),
                        hitsDC_->const_view().ranges(),
                        rangesDC_->view(),
                        ptCut_);

    auto nTotalMDs_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);
    auto nTotalMDs_buf_d = cms::alpakatools::make_device_view(queue_, rangesOccupancy.nTotalMDs());
    alpaka::memcpy(queue_, nTotalMDs_buf_h, nTotalMDs_buf_d);
    alpaka::wait(queue_);  // wait to get the data before manipulation

    *nTotalMDs_buf_h.data() += n_max_pixel_md_per_modules;
    unsigned int nTotalMDs = *nTotalMDs_buf_h.data();

    miniDoubletsDC_.emplace(queue_, static_cast<int32_t>(nTotalMDs), static_cast<int32_t>(nLowerModules_ + 1));

    auto mdsOccupancy = miniDoubletsDC_->view().miniDoubletsOccupancy();
    auto nMDs_view = cms::alpakatools::make_device_view(queue_, mdsOccupancy.nMDs());
    auto totOccupancyMDs_view = cms::alpakatools::make_device_view(queue_, mdsOccupancy.totOccupancyMDs());
    alpaka::memset(queue_, nMDs_view, 0u);
    alpaka::memset(queue_, totOccupancyMDs_view, 0u);
  }

  auto mdView = miniDoubletsDC_->view().miniDoublets();
  auto connView = cms::alpakatools::make_device_view(queue_, mdView.connectedMax());
  alpaka::memset(queue_, connView, 0u);

  unsigned int mdSize = pixelSize_ * 2;
  auto src_view_mdSize = cms::alpakatools::make_host_view(mdSize);

  auto mdsOccupancy = miniDoubletsDC_->view().miniDoubletsOccupancy();
  auto dst_view_nMDs = cms::alpakatools::make_device_view(queue_, mdsOccupancy.nMDs()[pixelModuleIndex_]);
  alpaka::memcpy(queue_, dst_view_nMDs, src_view_mdSize);

  auto dst_view_totOccupancyMDs =
      cms::alpakatools::make_device_view(queue_, mdsOccupancy.totOccupancyMDs()[pixelModuleIndex_]);
  alpaka::memcpy(queue_, dst_view_totOccupancyMDs, src_view_mdSize);

  alpaka::wait(queue_);  // FIXME: remove synch after inputs refactored to be in pinned memory

  constexpr int threadsPerBlockY = 16;
  auto const createMiniDoublets_workDiv =
      cms::alpakatools::make_workdiv<Acc2D>({nLowerModules_ / threadsPerBlockY, 1}, {threadsPerBlockY, 32});

  alpaka::exec<Acc2D>(queue_,
                      createMiniDoublets_workDiv,
                      CreateMiniDoublets{},
                      modules_.const_view().modules(),
                      lstInputDC_->const_view().hits(),
                      hitsDC_->const_view().extended(),
                      hitsDC_->const_view().ranges(),
                      miniDoubletsDC_->view().miniDoublets(),
                      miniDoubletsDC_->view().miniDoubletsOccupancy(),
                      rangesDC_->const_view(),
                      ptCut_,
                      clustSizeCut_);

  auto const addMiniDoubletRangesToEventExplicit_workDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 1024);

  alpaka::exec<Acc1D>(queue_,
                      addMiniDoubletRangesToEventExplicit_workDiv,
                      AddMiniDoubletRangesToEventExplicit{},
                      modules_.const_view().modules(),
                      miniDoubletsDC_->view().miniDoubletsOccupancy(),
                      rangesDC_->view(),
                      hitsDC_->const_view().ranges());

  if (addObjects_) {
    addMiniDoubletsToEventExplicit();
  }
}

void LSTEvent::createSegmentsWithModuleMap() {
  if (!segmentsDC_) {
    auto const countMDConn_wd = cms::alpakatools::make_workdiv<Acc3D>({nLowerModules_, 1, 1}, {1, 8, 32});

    alpaka::exec<Acc3D>(queue_,
                        countMDConn_wd,
                        CountMiniDoubletConnections{},
                        modules_.const_view().modules(),
                        miniDoubletsDC_->view().miniDoublets(),
                        miniDoubletsDC_->const_view().miniDoubletsOccupancy(),
                        rangesDC_->const_view(),
                        ptCut_);

    auto const createSegmentArrayRanges_workDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 1024);

    alpaka::exec<Acc1D>(queue_,
                        createSegmentArrayRanges_workDiv,
                        CreateSegmentArrayRanges{},
                        modules_.const_view().modules(),
                        rangesDC_->view(),
                        miniDoubletsDC_->const_view().miniDoublets(),
                        miniDoubletsDC_->const_view().miniDoubletsOccupancy());

    auto rangesOccupancy = rangesDC_->view();
    auto nTotalSegments_view_h = cms::alpakatools::make_host_view(nTotalSegments_);
    auto nTotalSegments_view_d = cms::alpakatools::make_device_view(queue_, rangesOccupancy.nTotalSegs());
    alpaka::memcpy(queue_, nTotalSegments_view_h, nTotalSegments_view_d);
    alpaka::wait(queue_);  // wait to get the value before manipulation

    nTotalSegments_ += n_max_pixel_segments_per_module;

    segmentsDC_.emplace(queue_, static_cast<int32_t>(nTotalSegments_), static_cast<int32_t>(nLowerModules_ + 1));

    auto segmentsOccupancy = segmentsDC_->view().segmentsOccupancy();
    auto segments = segmentsDC_->view().segments();
    auto nSegments_view = cms::alpakatools::make_device_view(queue_, segmentsOccupancy.nSegments());
    auto totOccupancySegments_view =
        cms::alpakatools::make_device_view(queue_, segmentsOccupancy.totOccupancySegments());
    alpaka::memset(queue_, nSegments_view, 0u);
    alpaka::memset(queue_, totOccupancySegments_view, 0u);
    auto conn_view = cms::alpakatools::make_device_view(queue_, segments.connectedMax());
    alpaka::memset(queue_, conn_view, 0u);

    auto src_view_size = cms::alpakatools::make_host_view(pixelSize_);

    auto dst_view_segments =
        cms::alpakatools::make_device_view(queue_, segmentsOccupancy.nSegments()[pixelModuleIndex_]);
    alpaka::memcpy(queue_, dst_view_segments, src_view_size);

    auto dst_view_totOccupancySegments =
        cms::alpakatools::make_device_view(queue_, segmentsOccupancy.totOccupancySegments()[pixelModuleIndex_]);
    alpaka::memcpy(queue_, dst_view_totOccupancySegments, src_view_size);
    alpaka::wait(queue_);
  }

  auto const createSegments_workDiv = cms::alpakatools::make_workdiv<Acc3D>({nLowerModules_, 1, 1}, {1, 8, 32});

  alpaka::exec<Acc3D>(queue_,
                      createSegments_workDiv,
                      CreateSegments{},
                      modules_.const_view().modules(),
                      miniDoubletsDC_->const_view().miniDoublets(),
                      miniDoubletsDC_->const_view().miniDoubletsOccupancy(),
                      segmentsDC_->view().segments(),
                      segmentsDC_->view().segmentsOccupancy(),
                      rangesDC_->const_view(),
                      ptCut_);

  auto const addSegmentRangesToEventExplicit_workDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 1024);

  alpaka::exec<Acc1D>(queue_,
                      addSegmentRangesToEventExplicit_workDiv,
                      AddSegmentRangesToEventExplicit{},
                      modules_.const_view().modules(),
                      segmentsDC_->view().segmentsOccupancy(),
                      rangesDC_->view());

  if (addObjects_) {
    addSegmentsToEventExplicit();
  }
}

void LSTEvent::createTriplets() {
  if (!tripletsDC_) {
    auto const countSegConn_wd = cms::alpakatools::make_workdiv<Acc3D>({nLowerModules_, 1, 1}, {1, 16, 16});

    alpaka::exec<Acc3D>(queue_,
                        countSegConn_wd,
                        CountSegmentConnections{},
                        modules_.const_view().modules(),
                        segmentsDC_->view().segments(),
                        segmentsDC_->const_view().segmentsOccupancy(),
                        rangesDC_->const_view());

    auto const createTripletArrayRanges_workDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 1024);

    alpaka::exec<Acc1D>(queue_,
                        createTripletArrayRanges_workDiv,
                        CreateTripletArrayRanges{},
                        modules_.const_view().modules(),
                        rangesDC_->view(),
                        segmentsDC_->const_view().segments(),
                        segmentsDC_->const_view().segmentsOccupancy());

    // TODO: Why are we pulling this back down only to put it back on the device in a new struct?
    auto rangesOccupancy = rangesDC_->view();
    auto maxTriplets_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);
    auto maxTriplets_buf_d = cms::alpakatools::make_device_view(queue_, rangesOccupancy.nTotalTrips());
    alpaka::memcpy(queue_, maxTriplets_buf_h, maxTriplets_buf_d);
    alpaka::wait(queue_);  // wait to get the value before using it

    tripletsDC_.emplace(queue_, static_cast<int32_t>(*maxTriplets_buf_h.data()), static_cast<int32_t>(nLowerModules_));

    auto tripletsOccupancy = tripletsDC_->view().tripletsOccupancy();
    auto nTriplets_view = cms::alpakatools::make_device_view(queue_, tripletsOccupancy.nTriplets());
    alpaka::memset(queue_, nTriplets_view, 0u);
    auto totOccupancyTriplets_view =
        cms::alpakatools::make_device_view(queue_, tripletsOccupancy.totOccupancyTriplets());
    alpaka::memset(queue_, totOccupancyTriplets_view, 0u);
    auto triplets = tripletsDC_->view().triplets();
    auto partOfPT5_view = cms::alpakatools::make_device_view(queue_, triplets.partOfPT5());
    alpaka::memset(queue_, partOfPT5_view, 0u);
    auto partOfT5_view = cms::alpakatools::make_device_view(queue_, triplets.partOfT5());
    alpaka::memset(queue_, partOfT5_view, 0u);
    auto partOfPT3_view = cms::alpakatools::make_device_view(queue_, triplets.partOfPT3());
    alpaka::memset(queue_, partOfPT3_view, 0u);
    auto connectedMax_view = cms::alpakatools::make_device_view(queue_, triplets.connectedMax());
    alpaka::memset(queue_, connectedMax_view, 0u);
    auto connectedLSMax_view =
        cms::alpakatools::make_device_view(queue_, triplets.connectedLSMax(), triplets.metadata().size());
    alpaka::memset(queue_, connectedLSMax_view, 0u);
  }

  uint16_t nonZeroModules = 0;
  unsigned int max_InnerSeg = 0;

  // Allocate and copy nSegments from device to host (only nLowerModules in OT, not the +1 with pLSs)
  auto nSegments_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, nLowerModules_);
  auto nSegments_buf_d = cms::alpakatools::make_device_view(
      queue_, segmentsDC_->const_view().segmentsOccupancy().nSegments(), nLowerModules_);
  alpaka::memcpy(queue_, nSegments_buf_h, nSegments_buf_d, nLowerModules_);

  // ... same for module_nConnectedModules
  // FIXME: replace by ES host data
  auto modules = modules_.const_view().modules();
  auto module_nConnectedModules_buf_h = cms::alpakatools::make_host_buffer<uint16_t[]>(queue_, nLowerModules_);
  auto module_nConnectedModules_buf_d =
      cms::alpakatools::make_device_view(queue_, modules.nConnectedModules(), nLowerModules_);  // only lower modules
  alpaka::memcpy(queue_, module_nConnectedModules_buf_h, module_nConnectedModules_buf_d, nLowerModules_);

  alpaka::wait(queue_);  // wait for nSegments and module_nConnectedModules before using

  auto const* nSegments = nSegments_buf_h.data();
  auto const* module_nConnectedModules = module_nConnectedModules_buf_h.data();

  // Allocate host index and fill it directly
  auto index_buf_h = cms::alpakatools::make_host_buffer<uint16_t[]>(queue_, nLowerModules_);
  auto* index = index_buf_h.data();

  for (uint16_t innerLowerModuleIndex = 0; innerLowerModuleIndex < nLowerModules_; innerLowerModuleIndex++) {
    uint16_t nConnectedModules = module_nConnectedModules[innerLowerModuleIndex];
    unsigned int nInnerSegments = nSegments[innerLowerModuleIndex];
    if (nConnectedModules != 0 and nInnerSegments != 0) {
      index[nonZeroModules] = innerLowerModuleIndex;
      nonZeroModules++;
    }
    max_InnerSeg = std::max(max_InnerSeg, nInnerSegments);
  }

  // Allocate and copy to device index
  auto index_gpu_buf = cms::alpakatools::make_device_buffer<uint16_t[]>(queue_, nLowerModules_);
  alpaka::memcpy(queue_, index_gpu_buf, index_buf_h, nonZeroModules);

  auto const createTriplets_workDiv = cms::alpakatools::make_workdiv<Acc3D>({nonZeroModules, 1, 1}, {1, 16, 16});

  alpaka::exec<Acc3D>(queue_,
                      createTriplets_workDiv,
                      CreateTriplets{},
                      modules_.const_view().modules(),
                      miniDoubletsDC_->const_view().miniDoublets(),
                      segmentsDC_->const_view().segments(),
                      segmentsDC_->const_view().segmentsOccupancy(),
                      tripletsDC_->view().triplets(),
                      tripletsDC_->view().tripletsOccupancy(),
                      rangesDC_->const_view(),
                      index_gpu_buf.data(),
                      nonZeroModules,
                      ptCut_);

  auto const addTripletRangesToEventExplicit_workDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 1024);

  if (nonZeroModules != 0) {
    alpaka::exec<Acc1D>(queue_,
                        addTripletRangesToEventExplicit_workDiv,
                        AddTripletRangesToEventExplicit{},
                        modules_.const_view().modules(),
                        tripletsDC_->const_view().tripletsOccupancy(),
                        rangesDC_->view());

    if (addObjects_) {
      addTripletsToEventExplicit();
    }
  }
}

void LSTEvent::createTrackCandidates(bool no_pls_dupclean, bool tc_pls_triplets) {
  if (!trackCandidatesBaseDC_) {
    trackCandidatesBaseDC_.emplace(n_max_nonpixel_track_candidates + n_max_pixel_track_candidates, queue_);
    trackCandidatesBaseDC_->zeroInitialise(queue_);

    trackCandidatesExtendedDC_.emplace(n_max_nonpixel_track_candidates + n_max_pixel_track_candidates, queue_);
    trackCandidatesExtendedDC_->zeroInitialise(queue_);
  }

  auto const crossCleanpT3_workDiv = cms::alpakatools::make_workdiv<Acc2D>({20, 4}, {64, 16});

  alpaka::exec<Acc2D>(queue_,
                      crossCleanpT3_workDiv,
                      CrossCleanpT3{},
                      modules_.const_view().modules(),
                      rangesDC_->const_view(),
                      pixelTripletsDC_->view(),
                      lstInputDC_->const_view().pixelSeeds(),
                      pixelQuintupletsDC_->const_view());

  auto const addpT3asTrackCandidates_workDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 512);

  alpaka::exec<Acc1D>(queue_,
                      addpT3asTrackCandidates_workDiv,
                      AddpT3asTrackCandidates{},
                      nLowerModules_,
                      pixelTripletsDC_->const_view(),
                      trackCandidatesBaseDC_->view(),
                      trackCandidatesExtendedDC_->view(),
                      lstInputDC_->const_view().pixelSeeds(),
                      rangesDC_->const_view());

  // Pull nEligibleT5Modules from the device.
  auto rangesOccupancy = rangesDC_->view();
  auto nEligibleModules_buf_h = cms::alpakatools::make_host_buffer<uint16_t>(queue_);
  auto nEligibleModules_buf_d = cms::alpakatools::make_device_view(queue_, rangesOccupancy.nEligibleT5Modules());
  alpaka::memcpy(queue_, nEligibleModules_buf_h, nEligibleModules_buf_d);
  alpaka::wait(queue_);  // wait to get the value before using
  auto const nEligibleModules = *nEligibleModules_buf_h.data();

  constexpr int threadsPerBlockY = 16;
  constexpr int threadsPerBlockX = 32;
  auto const removeDupQuintupletsBeforeTC_workDiv = cms::alpakatools::make_workdiv<Acc2D>(
      {std::max(nEligibleModules / threadsPerBlockY, 1), std::max(nEligibleModules / threadsPerBlockX, 1)}, {16, 32});

  alpaka::exec<Acc2D>(queue_,
                      removeDupQuintupletsBeforeTC_workDiv,
                      RemoveDupQuintupletsBeforeTC{},
                      quintupletsDC_->view().quintuplets(),
                      quintupletsDC_->view().quintupletsOccupancy(),
                      rangesDC_->const_view());

  constexpr int threadsPerBlock = 32;
  auto const crossCleanT5_workDiv = cms::alpakatools::make_workdiv<Acc3D>(
      {(nLowerModules_ / threadsPerBlock) + 1, 1, max_blocks}, {threadsPerBlock, 1, threadsPerBlock});

  alpaka::exec<Acc3D>(queue_,
                      crossCleanT5_workDiv,
                      CrossCleanT5{},
                      modules_.const_view().modules(),
                      quintupletsDC_->view().quintuplets(),
                      quintupletsDC_->const_view().quintupletsOccupancy(),
                      pixelQuintupletsDC_->const_view(),
                      pixelTripletsDC_->const_view(),
                      rangesDC_->const_view());

  auto const addT5asTrackCandidate_workDiv = cms::alpakatools::make_workdiv<Acc2D>({8, 10}, {8, 128});

  alpaka::exec<Acc2D>(queue_,
                      addT5asTrackCandidate_workDiv,
                      AddT5asTrackCandidate{},
                      nLowerModules_,
                      quintupletsDC_->const_view().quintuplets(),
                      quintupletsDC_->const_view().quintupletsOccupancy(),
                      trackCandidatesBaseDC_->view(),
                      trackCandidatesExtendedDC_->view(),
                      rangesDC_->const_view());

  auto nEligibleModulesT4_buf_h = cms::alpakatools::make_host_buffer<uint16_t>(queue_);
  auto nEligibleModulesT4_buf_d = cms::alpakatools::make_device_view(queue_, rangesOccupancy.nEligibleT4Modules());
  alpaka::memcpy(queue_, nEligibleModulesT4_buf_h, nEligibleModulesT4_buf_d);
  alpaka::wait(queue_);  // wait to get the value before using
  auto const nEligibleModulesT4 = *nEligibleModulesT4_buf_h.data();

  auto const removeDupQuadrupletsBeforeTC_workDiv = cms::alpakatools::make_workdiv<Acc2D>(
      {std::max(nEligibleModulesT4 / threadsPerBlockY, 1), std::max(nEligibleModulesT4 / threadsPerBlockX, 1)},
      {16, 32});

  alpaka::exec<Acc2D>(queue_,
                      removeDupQuadrupletsBeforeTC_workDiv,
                      RemoveDupQuadrupletsBeforeTC{},
                      quadrupletsDC_->view().quadruplets(),
                      quadrupletsDC_->view().quadrupletsOccupancy(),
                      rangesDC_->const_view());

  auto const crossCleanT4_workDiv = cms::alpakatools::make_workdiv<Acc3D>(
      {(nLowerModules_ / threadsPerBlock) + 1, 1, max_blocks}, {threadsPerBlock, 1, threadsPerBlock});

  alpaka::exec<Acc3D>(queue_,
                      crossCleanT4_workDiv,
                      CrossCleanT4{},
                      modules_.const_view().modules(),
                      quadrupletsDC_->view().quadruplets(),
                      quadrupletsDC_->const_view().quadrupletsOccupancy(),
                      pixelQuintupletsDC_->const_view(),
                      pixelTripletsDC_->const_view(),
                      quintupletsDC_->const_view().quintuplets(),
                      trackCandidatesBaseDC_->view(),
                      trackCandidatesExtendedDC_->view(),
                      miniDoubletsDC_->view().miniDoublets(),
                      segmentsDC_->view().segments(),
                      tripletsDC_->view().triplets(),
                      rangesDC_->const_view());

  auto const addT4asTrackCandidate_workDiv = cms::alpakatools::make_workdiv<Acc2D>({8, 10}, {8, 128});

  alpaka::exec<Acc2D>(queue_,
                      addT4asTrackCandidate_workDiv,
                      AddT4asTrackCandidate{},
                      nLowerModules_,
                      quadrupletsDC_->view().quadruplets(),
                      quadrupletsDC_->const_view().quadrupletsOccupancy(),
                      tripletsDC_->const_view().triplets(),
                      trackCandidatesBaseDC_->view(),
                      trackCandidatesExtendedDC_->view(),
                      rangesDC_->const_view());

  if (!no_pls_dupclean) {
    auto const checkHitspLS_workDiv = cms::alpakatools::make_workdiv<Acc2D>({max_blocks * 4, max_blocks / 4}, {16, 16});

    alpaka::exec<Acc2D>(queue_,
                        checkHitspLS_workDiv,
                        CheckHitspLS{},
                        modules_.const_view().modules(),
                        segmentsDC_->const_view().segmentsOccupancy(),
                        lstInputDC_->const_view().pixelSeeds(),
                        pixelSegmentsDC_->view(),
                        true);
  }

  auto const crossCleanpLS_workDiv = cms::alpakatools::make_workdiv<Acc2D>({20, 4}, {32, 16});

  alpaka::exec<Acc2D>(queue_,
                      crossCleanpLS_workDiv,
                      CrossCleanpLS{},
                      modules_.const_view().modules(),
                      rangesDC_->const_view(),
                      pixelTripletsDC_->const_view(),
                      trackCandidatesBaseDC_->view(),
                      trackCandidatesExtendedDC_->view(),
                      segmentsDC_->const_view().segments(),
                      segmentsDC_->const_view().segmentsOccupancy(),
                      lstInputDC_->const_view().pixelSeeds(),
                      pixelSegmentsDC_->view(),
                      miniDoubletsDC_->const_view().miniDoublets(),
                      lstInputDC_->const_view().hits(),
                      quintupletsDC_->const_view().quintuplets(),
                      quadrupletsDC_->const_view().quadruplets());

  auto const addpLSasTrackCandidate_workDiv = cms::alpakatools::make_workdiv<Acc1D>(max_blocks, 384);

  alpaka::exec<Acc1D>(queue_,
                      addpLSasTrackCandidate_workDiv,
                      AddpLSasTrackCandidate{},
                      nLowerModules_,
                      trackCandidatesBaseDC_->view(),
                      trackCandidatesExtendedDC_->view(),
                      segmentsDC_->const_view().segmentsOccupancy(),
                      lstInputDC_->const_view().pixelSeeds(),
                      pixelSegmentsDC_->const_view(),
                      tc_pls_triplets);

  // Get number of TCs to configure grid
  auto nTrackCandidates_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);
  auto nTrackCandidates_buf_d =
      cms::alpakatools::make_device_view(queue_, (*trackCandidatesBaseDC_)->nTrackCandidates());
  alpaka::memcpy(queue_, nTrackCandidates_buf_h, nTrackCandidates_buf_d);
  alpaka::wait(queue_);
  unsigned int nTC = *nTrackCandidates_buf_h.data();

  auto const wd = cms::alpakatools::make_workdiv<Acc1D>(nTC, 128);

  alpaka::exec<Acc1D>(queue_,
                      wd,
                      ExtendTrackCandidatesFromDupT5{},
                      modules_.const_view().modules(),
                      rangesDC_->const_view(),
                      quintupletsDC_->const_view().quintuplets(),
                      quintupletsDC_->const_view().quintupletsOccupancy(),
                      trackCandidatesBaseDC_->view(),
                      trackCandidatesExtendedDC_->view());

  // Check if either n_max_pixel_track_candidates or n_max_nonpixel_track_candidates was reached
  auto nTrackCanpT5Host_buf = cms::alpakatools::make_host_buffer<unsigned int>(queue_);
  auto nTrackCanpT3Host_buf = cms::alpakatools::make_host_buffer<unsigned int>(queue_);
  auto nTrackCanpLSHost_buf = cms::alpakatools::make_host_buffer<unsigned int>(queue_);
  auto nTrackCanT5Host_buf = cms::alpakatools::make_host_buffer<unsigned int>(queue_);
  auto nTrackCanT4Host_buf = cms::alpakatools::make_host_buffer<unsigned int>(queue_);
  alpaka::memcpy(queue_,
                 nTrackCanpT5Host_buf,
                 cms::alpakatools::make_device_view(queue_, (*trackCandidatesExtendedDC_)->nTrackCandidatespT5()));
  alpaka::memcpy(queue_,
                 nTrackCanpT3Host_buf,
                 cms::alpakatools::make_device_view(queue_, (*trackCandidatesExtendedDC_)->nTrackCandidatespT3()));
  alpaka::memcpy(queue_,
                 nTrackCanpLSHost_buf,
                 cms::alpakatools::make_device_view(queue_, (*trackCandidatesExtendedDC_)->nTrackCandidatespLS()));
  alpaka::memcpy(queue_,
                 nTrackCanT5Host_buf,
                 cms::alpakatools::make_device_view(queue_, (*trackCandidatesExtendedDC_)->nTrackCandidatesT5()));
  alpaka::memcpy(queue_,
                 nTrackCanT4Host_buf,
                 cms::alpakatools::make_device_view(queue_, (*trackCandidatesExtendedDC_)->nTrackCandidatesT4()));
  alpaka::wait(queue_);  // wait to get the values before using them

  auto nTrackCandidatespT5 = *nTrackCanpT5Host_buf.data();
  auto nTrackCandidatespT3 = *nTrackCanpT3Host_buf.data();
  auto nTrackCandidatespLS = *nTrackCanpLSHost_buf.data();
  auto nTrackCandidatesT5 = *nTrackCanT5Host_buf.data();
  auto nTrackCandidatesT4 = *nTrackCanT4Host_buf.data();
  if ((nTrackCandidatespT5 + nTrackCandidatespT3 + nTrackCandidatespLS == n_max_pixel_track_candidates) ||
      (nTrackCandidatesT5 + nTrackCandidatesT4 == n_max_nonpixel_track_candidates)) {
    lstWarning(
        "\
        ****************************************************************************************************\n\
        * Track candidates were possibly truncated.                                                        *\n\
        * You may need to increase either n_max_pixel_track_candidates or n_max_nonpixel_track_candidates. *\n\
        * Run the code with the WARNINGS flag activated for more details.                                  *\n\
        ****************************************************************************************************");
  }
}

void LSTEvent::createPixelTriplets() {
  if (!pixelTripletsDC_) {
    pixelTripletsDC_.emplace(n_max_pixel_triplets, queue_);
    auto nPixelTriplets_view = cms::alpakatools::make_device_view(queue_, (*pixelTripletsDC_)->nPixelTriplets());
    alpaka::memset(queue_, nPixelTriplets_view, 0u);
    auto totOccupancyPixelTriplets_view =
        cms::alpakatools::make_device_view(queue_, (*pixelTripletsDC_)->totOccupancyPixelTriplets());
    alpaka::memset(queue_, totOccupancyPixelTriplets_view, 0u);
  }
  SegmentsOccupancy segmentsOccupancy = segmentsDC_->view().segmentsOccupancy();
  PixelSeedsConst pixelSeeds = lstInputDC_->const_view().pixelSeeds();

  auto superbins_buf = cms::alpakatools::make_host_buffer<int[]>(queue_, n_max_pixel_segments_per_module);
  auto pixelTypes_buf = cms::alpakatools::make_host_buffer<PixelType[]>(queue_, n_max_pixel_segments_per_module);

  alpaka::memcpy(queue_, superbins_buf, cms::alpakatools::make_device_view(queue_, pixelSeeds.superbin(), pixelSize_));
  alpaka::memcpy(
      queue_, pixelTypes_buf, cms::alpakatools::make_device_view(queue_, pixelSeeds.pixelType(), pixelSize_));
  auto const* superbins = superbins_buf.data();
  auto const* pixelTypes = pixelTypes_buf.data();

  unsigned int nInnerSegments;
  auto nInnerSegments_src_view = cms::alpakatools::make_host_view(nInnerSegments);

  // Create a sub-view for the device buffer
  auto dev_view_nSegments = cms::alpakatools::make_device_view(queue_, segmentsOccupancy.nSegments()[nLowerModules_]);

  alpaka::memcpy(queue_, nInnerSegments_src_view, dev_view_nSegments);
  alpaka::wait(queue_);  // wait to get nInnerSegments (also superbins and pixelTypes) before using

  auto connectedPixelSize_host_buf = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, nInnerSegments);
  auto connectedPixelIndex_host_buf = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, nInnerSegments);
  auto connectedPixelSize_dev_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, nInnerSegments);
  auto connectedPixelIndex_dev_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, nInnerSegments);

  unsigned int* connectedPixelSize_host = connectedPixelSize_host_buf.data();
  unsigned int* connectedPixelIndex_host = connectedPixelIndex_host_buf.data();

  int pixelIndexOffsetPos =
      pixelMapping_.connectedPixelsIndex[size_superbins - 1] + pixelMapping_.connectedPixelsSizes[size_superbins - 1];
  int pixelIndexOffsetNeg = pixelMapping_.connectedPixelsIndexPos[size_superbins - 1] +
                            pixelMapping_.connectedPixelsSizesPos[size_superbins - 1] + pixelIndexOffsetPos;

  // TODO: check if a map/reduction to just eligible pLSs would speed up the kernel
  // the current selection still leaves a significant fraction of unmatchable pLSs
  for (unsigned int i = 0; i < nInnerSegments; i++) {  // loop over # pLS
    PixelType pixelType = pixelTypes[i];               // Get pixel type for this pLS
    int superbin = superbins[i];                       // Get superbin for this pixel
    if ((superbin < 0) or (superbin >= (int)size_superbins) or
        ((pixelType != PixelType::kHighPt) and (pixelType != PixelType::kLowPtPosCurv) and
         (pixelType != PixelType::kLowPtNegCurv))) {
      connectedPixelSize_host[i] = 0;
      connectedPixelIndex_host[i] = 0;
      continue;
    }

    // Used pixel type to select correct size-index arrays
    switch (pixelType) {
      case PixelType::kInvalid:
        break;
      case PixelType::kHighPt:
        // number of connected modules to this pixel
        connectedPixelSize_host[i] = pixelMapping_.connectedPixelsSizes[superbin];
        // index to get start of connected modules for this superbin in map
        connectedPixelIndex_host[i] = pixelMapping_.connectedPixelsIndex[superbin];
        break;
      case PixelType::kLowPtPosCurv:
        // number of connected modules to this pixel
        connectedPixelSize_host[i] = pixelMapping_.connectedPixelsSizesPos[superbin];
        // index to get start of connected modules for this superbin in map
        connectedPixelIndex_host[i] = pixelMapping_.connectedPixelsIndexPos[superbin] + pixelIndexOffsetPos;
        break;
      case PixelType::kLowPtNegCurv:
        // number of connected modules to this pixel
        connectedPixelSize_host[i] = pixelMapping_.connectedPixelsSizesNeg[superbin];
        // index to get start of connected modules for this superbin in map
        connectedPixelIndex_host[i] = pixelMapping_.connectedPixelsIndexNeg[superbin] + pixelIndexOffsetNeg;
        break;
    }
  }

  alpaka::memcpy(queue_, connectedPixelSize_dev_buf, connectedPixelSize_host_buf, nInnerSegments);
  alpaka::memcpy(queue_, connectedPixelIndex_dev_buf, connectedPixelIndex_host_buf, nInnerSegments);

  auto const createPixelTripletsFromMap_workDiv =
      cms::alpakatools::make_workdiv<Acc3D>({4096, 16 /* above median of connected modules*/, 1}, {4, 1, 32});

  alpaka::exec<Acc3D>(queue_,
                      createPixelTripletsFromMap_workDiv,
                      CreatePixelTripletsFromMap{},
                      modules_.const_view().modules(),
                      modules_.const_view().modulesPixel(),
                      rangesDC_->const_view(),
                      miniDoubletsDC_->const_view().miniDoublets(),
                      segmentsDC_->const_view().segments(),
                      lstInputDC_->const_view().pixelSeeds(),
                      pixelSegmentsDC_->const_view(),
                      tripletsDC_->view().triplets(),
                      tripletsDC_->const_view().tripletsOccupancy(),
                      pixelTripletsDC_->view(),
                      connectedPixelSize_dev_buf.data(),
                      connectedPixelIndex_dev_buf.data(),
                      nInnerSegments,
                      ptCut_);

#ifdef WARNINGS
  auto nPixelTriplets_buf = cms::alpakatools::make_host_buffer<unsigned int>(queue_);

  alpaka::memcpy(
      queue_, nPixelTriplets_buf, cms::alpakatools::make_device_view(queue_, (*pixelTripletsDC_)->nPixelTriplets()));
  alpaka::wait(queue_);  // wait to get the value before using it

  std::cout << "number of pixel triplets = " << *nPixelTriplets_buf.data() << std::endl;
#endif

  //pT3s can be cleaned here because they're not used in making pT5s!
  //seems like more blocks lead to conflicting writes
  auto const removeDupPixelTripletsFromMap_workDiv = cms::alpakatools::make_workdiv<Acc2D>({40, 1}, {16, 16});

  alpaka::exec<Acc2D>(
      queue_, removeDupPixelTripletsFromMap_workDiv, RemoveDupPixelTripletsFromMap{}, pixelTripletsDC_->view());
}

void LSTEvent::createQuintuplets() {
  auto const countConn_workDiv = cms::alpakatools::make_workdiv<Acc3D>({nLowerModules_, 1, 1}, {1, 8, 32});

  alpaka::exec<Acc3D>(queue_,
                      countConn_workDiv,
                      CountTripletConnections{},
                      modules_.const_view().modules(),
                      miniDoubletsDC_->const_view().miniDoublets(),
                      segmentsDC_->const_view().segments(),
                      tripletsDC_->view().triplets(),
                      tripletsDC_->const_view().tripletsOccupancy(),
                      rangesDC_->const_view(),
                      ptCut_);

  auto const createEligibleModulesListForQuintuplets_workDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 1024);

  alpaka::exec<Acc1D>(queue_,
                      createEligibleModulesListForQuintuplets_workDiv,
                      CreateEligibleModulesListForQuintuplets{},
                      modules_.const_view().modules(),
                      tripletsDC_->const_view().tripletsOccupancy(),
                      rangesDC_->view(),
                      tripletsDC_->view().triplets());

  auto nEligibleT5Modules_buf = cms::alpakatools::make_host_buffer<uint16_t>(queue_);
  auto nTotalQuintuplets_buf = cms::alpakatools::make_host_buffer<unsigned int>(queue_);
  auto rangesOccupancy = rangesDC_->view();
  auto nEligibleT5Modules_view_d = cms::alpakatools::make_device_view(queue_, rangesOccupancy.nEligibleT5Modules());
  auto nTotalQuintuplets_view_d = cms::alpakatools::make_device_view(queue_, rangesOccupancy.nTotalQuints());
  alpaka::memcpy(queue_, nEligibleT5Modules_buf, nEligibleT5Modules_view_d);
  alpaka::memcpy(queue_, nTotalQuintuplets_buf, nTotalQuintuplets_view_d);
  alpaka::wait(queue_);  // wait for the values before using them

  auto nEligibleT5Modules = *nEligibleT5Modules_buf.data();
  auto nTotalQuintuplets = *nTotalQuintuplets_buf.data();

  if (!quintupletsDC_) {
    quintupletsDC_.emplace(queue_, static_cast<int32_t>(nTotalQuintuplets), static_cast<int32_t>(nLowerModules_));
    auto quintupletsOccupancy = quintupletsDC_->view().quintupletsOccupancy();
    auto nQuintuplets_view = cms::alpakatools::make_device_view(queue_, quintupletsOccupancy.nQuintuplets());
    alpaka::memset(queue_, nQuintuplets_view, 0u);
    auto totOccupancyQuintuplets_view =
        cms::alpakatools::make_device_view(queue_, quintupletsOccupancy.totOccupancyQuintuplets());
    alpaka::memset(queue_, totOccupancyQuintuplets_view, 0u);
    auto quintuplets = quintupletsDC_->view().quintuplets();
    auto isDup_view = cms::alpakatools::make_device_view(queue_, quintuplets.isDup());
    alpaka::memset(queue_, isDup_view, 0u);
    auto tightCutFlag_view = cms::alpakatools::make_device_view(queue_, quintuplets.tightCutFlag());
    alpaka::memset(queue_, tightCutFlag_view, 0u);
    auto partOfPT5_view = cms::alpakatools::make_device_view(queue_, quintuplets.partOfPT5());
    alpaka::memset(queue_, partOfPT5_view, 0u);
  }

  auto const createQuintuplets_workDiv =
      cms::alpakatools::make_workdiv<Acc3D>({std::max((int)nEligibleT5Modules, 1), 1, 1}, {1, 8, 32});

  alpaka::exec<Acc3D>(queue_,
                      createQuintuplets_workDiv,
                      CreateQuintuplets{},
                      modules_.const_view().modules(),
                      miniDoubletsDC_->const_view().miniDoublets(),
                      segmentsDC_->const_view().segments(),
                      tripletsDC_->view().triplets(),
                      tripletsDC_->const_view().tripletsOccupancy(),
                      quintupletsDC_->view().quintuplets(),
                      quintupletsDC_->view().quintupletsOccupancy(),
                      rangesDC_->const_view(),
                      nEligibleT5Modules,
                      ptCut_);

  auto const removeDupQuintupletsAfterBuild_workDiv =
      cms::alpakatools::make_workdiv<Acc3D>({max_blocks, 1, 1}, {1, 16, 16});

  alpaka::exec<Acc3D>(queue_,
                      removeDupQuintupletsAfterBuild_workDiv,
                      RemoveDupQuintupletsAfterBuild{},
                      modules_.const_view().modules(),
                      quintupletsDC_->view().quintuplets(),
                      quintupletsDC_->const_view().quintupletsOccupancy(),
                      rangesDC_->const_view());

  auto const addQuintupletRangesToEventExplicit_workDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 1024);

  alpaka::exec<Acc1D>(queue_,
                      addQuintupletRangesToEventExplicit_workDiv,
                      AddQuintupletRangesToEventExplicit{},
                      modules_.const_view().modules(),
                      quintupletsDC_->const_view().quintupletsOccupancy(),
                      rangesDC_->view());

  if (addObjects_) {
    addQuintupletsToEventExplicit();
  }
}

void LSTEvent::pixelLineSegmentCleaning(bool no_pls_dupclean) {
  if (!no_pls_dupclean) {
    auto const checkHitspLS_workDiv = cms::alpakatools::make_workdiv<Acc2D>({max_blocks * 4, max_blocks / 4}, {16, 16});

    alpaka::exec<Acc2D>(queue_,
                        checkHitspLS_workDiv,
                        CheckHitspLS{},
                        modules_.const_view().modules(),
                        segmentsDC_->const_view().segmentsOccupancy(),
                        lstInputDC_->const_view().pixelSeeds(),
                        pixelSegmentsDC_->view(),
                        false);
  }
}

void LSTEvent::createPixelQuintuplets() {
  if (!pixelQuintupletsDC_) {
    pixelQuintupletsDC_.emplace(n_max_pixel_quintuplets, queue_);
    auto nPixelQuintuplets_view =
        cms::alpakatools::make_device_view(queue_, (*pixelQuintupletsDC_)->nPixelQuintuplets());
    alpaka::memset(queue_, nPixelQuintuplets_view, 0u);
    auto totOccupancyPixelQuintuplets_view =
        cms::alpakatools::make_device_view(queue_, (*pixelQuintupletsDC_)->totOccupancyPixelQuintuplets());
    alpaka::memset(queue_, totOccupancyPixelQuintuplets_view, 0u);
  }
  if (!trackCandidatesBaseDC_) {
    trackCandidatesBaseDC_.emplace(n_max_nonpixel_track_candidates + n_max_pixel_track_candidates, queue_);
    trackCandidatesBaseDC_->zeroInitialise(queue_);

    trackCandidatesExtendedDC_.emplace(n_max_nonpixel_track_candidates + n_max_pixel_track_candidates, queue_);
    trackCandidatesExtendedDC_->zeroInitialise(queue_);
  }
  SegmentsOccupancy segmentsOccupancy = segmentsDC_->view().segmentsOccupancy();
  PixelSeedsConst pixelSeeds = lstInputDC_->const_view().pixelSeeds();

  auto superbins_buf = cms::alpakatools::make_host_buffer<int[]>(queue_, n_max_pixel_segments_per_module);
  auto pixelTypes_buf = cms::alpakatools::make_host_buffer<PixelType[]>(queue_, n_max_pixel_segments_per_module);

  alpaka::memcpy(queue_, superbins_buf, cms::alpakatools::make_device_view(queue_, pixelSeeds.superbin(), pixelSize_));
  alpaka::memcpy(
      queue_, pixelTypes_buf, cms::alpakatools::make_device_view(queue_, pixelSeeds.pixelType(), pixelSize_));
  auto const* superbins = superbins_buf.data();
  auto const* pixelTypes = pixelTypes_buf.data();

  unsigned int nInnerSegments;
  auto nInnerSegments_src_view = cms::alpakatools::make_host_view(nInnerSegments);

  // Create a sub-view for the device buffer
  unsigned int totalModules = nLowerModules_ + 1;
  auto dev_view_nSegments_buf = cms::alpakatools::make_device_view(queue_, segmentsOccupancy.nSegments(), totalModules);
  auto dev_view_nSegments = cms::alpakatools::make_device_view(queue_, segmentsOccupancy.nSegments()[nLowerModules_]);

  alpaka::memcpy(queue_, nInnerSegments_src_view, dev_view_nSegments);
  alpaka::wait(queue_);  // wait to get nInnerSegments (also superbins and pixelTypes) before using

  auto connectedPixelSize_host_buf = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, nInnerSegments);
  auto connectedPixelIndex_host_buf = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, nInnerSegments);
  auto connectedPixelSize_dev_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, nInnerSegments);
  auto connectedPixelIndex_dev_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, nInnerSegments);

  auto* connectedPixelSize_host = connectedPixelSize_host_buf.data();
  auto* connectedPixelIndex_host = connectedPixelIndex_host_buf.data();

  int pixelIndexOffsetPos = pixelMapping_.connectedPixelsIndex[::size_superbins - 1] +
                            pixelMapping_.connectedPixelsSizes[::size_superbins - 1];
  int pixelIndexOffsetNeg = pixelMapping_.connectedPixelsIndexPos[::size_superbins - 1] +
                            pixelMapping_.connectedPixelsSizesPos[::size_superbins - 1] + pixelIndexOffsetPos;

  // Loop over # pLS
  for (unsigned int i = 0; i < nInnerSegments; i++) {
    PixelType pixelType = pixelTypes[i];  // Get pixel type for this pLS
    int superbin = superbins[i];          // Get superbin for this pixel
    if ((superbin < 0) or (superbin >= (int)size_superbins) or
        ((pixelType != PixelType::kHighPt) and (pixelType != PixelType::kLowPtPosCurv) and
         (pixelType != PixelType::kLowPtNegCurv))) {
      connectedPixelSize_host[i] = 0;
      connectedPixelIndex_host[i] = 0;
      continue;
    }

    // Used pixel type to select correct size-index arrays
    switch (pixelType) {
      case PixelType::kInvalid:
        break;
      case PixelType::kHighPt:
        // number of connected modules to this pixel
        connectedPixelSize_host[i] = pixelMapping_.connectedPixelsSizes[superbin];
        // index to get start of connected modules for this superbin in map
        connectedPixelIndex_host[i] = pixelMapping_.connectedPixelsIndex[superbin];
        break;
      case PixelType::kLowPtPosCurv:
        // number of connected modules to this pixel
        connectedPixelSize_host[i] = pixelMapping_.connectedPixelsSizesPos[superbin];
        // index to get start of connected modules for this superbin in map
        connectedPixelIndex_host[i] = pixelMapping_.connectedPixelsIndexPos[superbin] + pixelIndexOffsetPos;
        break;
      case PixelType::kLowPtNegCurv:
        // number of connected modules to this pixel
        connectedPixelSize_host[i] = pixelMapping_.connectedPixelsSizesNeg[superbin];
        // index to get start of connected modules for this superbin in map
        connectedPixelIndex_host[i] = pixelMapping_.connectedPixelsIndexNeg[superbin] + pixelIndexOffsetNeg;
        break;
    }
  }

  alpaka::memcpy(queue_, connectedPixelSize_dev_buf, connectedPixelSize_host_buf, nInnerSegments);
  alpaka::memcpy(queue_, connectedPixelIndex_dev_buf, connectedPixelIndex_host_buf, nInnerSegments);

  auto const createPixelQuintupletsFromMap_workDiv =
      cms::alpakatools::make_workdiv<Acc3D>({max_blocks, 16, 1}, {16, 1, 16});

  alpaka::exec<Acc3D>(queue_,
                      createPixelQuintupletsFromMap_workDiv,
                      CreatePixelQuintupletsFromMap{},
                      modules_.const_view().modules(),
                      modules_.const_view().modulesPixel(),
                      miniDoubletsDC_->const_view().miniDoublets(),
                      segmentsDC_->const_view().segments(),
                      lstInputDC_->const_view().pixelSeeds(),
                      pixelSegmentsDC_->view(),
                      tripletsDC_->view().triplets(),
                      quintupletsDC_->view().quintuplets(),
                      quintupletsDC_->const_view().quintupletsOccupancy(),
                      pixelQuintupletsDC_->view(),
                      connectedPixelSize_dev_buf.data(),
                      connectedPixelIndex_dev_buf.data(),
                      nInnerSegments,
                      rangesDC_->const_view(),
                      ptCut_);

  auto const removeDupPixelQuintupletsFromMap_workDiv =
      cms::alpakatools::make_workdiv<Acc2D>({max_blocks, 1}, {16, 16});

  alpaka::exec<Acc2D>(queue_,
                      removeDupPixelQuintupletsFromMap_workDiv,
                      RemoveDupPixelQuintupletsFromMap{},
                      pixelQuintupletsDC_->view());

  auto const addpT5asTrackCandidate_workDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 256);

  alpaka::exec<Acc1D>(queue_,
                      addpT5asTrackCandidate_workDiv,
                      AddpT5asTrackCandidate{},
                      nLowerModules_,
                      pixelQuintupletsDC_->const_view(),
                      trackCandidatesBaseDC_->view(),
                      trackCandidatesExtendedDC_->view(),
                      lstInputDC_->const_view().pixelSeeds(),
                      rangesDC_->const_view());

#ifdef WARNINGS
  auto nPixelQuintuplets_buf = cms::alpakatools::make_host_buffer<unsigned int>(queue_);

  alpaka::memcpy(queue_,
                 nPixelQuintuplets_buf,
                 cms::alpakatools::make_device_view(queue_, (*pixelQuintupletsDC_)->nPixelQuintuplets()));
  alpaka::wait(queue_);  // wait to get the value before using it

  std::cout << "number of pixel quintuplets = " << *nPixelQuintuplets_buf.data() << std::endl;
#endif
}

void LSTEvent::createQuadruplets() {
  auto const countLSConn_workDiv = cms::alpakatools::make_workdiv<Acc3D>({nLowerModules_, 1, 1}, {1, 8, 32});

  alpaka::exec<Acc3D>(queue_,
                      countLSConn_workDiv,
                      CountTripletLSConnections{},
                      modules_.const_view().modules(),
                      miniDoubletsDC_->const_view().miniDoublets(),
                      segmentsDC_->const_view().segments(),
                      tripletsDC_->view().triplets(),
                      tripletsDC_->const_view().tripletsOccupancy(),
                      rangesDC_->const_view(),
                      ptCut_);

  auto const createEligibleModulesListForQuadruplets_workDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 1024);

  alpaka::exec<Acc1D>(queue_,
                      createEligibleModulesListForQuadruplets_workDiv,
                      CreateEligibleModulesListForQuadruplets{},
                      modules_.const_view().modules(),
                      tripletsDC_->const_view().tripletsOccupancy(),
                      rangesDC_->view(),
                      tripletsDC_->view().triplets());

  auto nEligibleT4Modules_buf = cms::alpakatools::make_host_buffer<uint16_t>(queue_);
  auto nTotalQuadruplets_buf = cms::alpakatools::make_host_buffer<unsigned int>(queue_);
  auto rangesOccupancy = rangesDC_->view();
  auto nEligibleT4Modules_view_d = cms::alpakatools::make_device_view(queue_, rangesOccupancy.nEligibleT4Modules());
  auto nTotalQuadruplets_view_d = cms::alpakatools::make_device_view(queue_, rangesOccupancy.nTotalQuads());
  alpaka::memcpy(queue_, nEligibleT4Modules_buf, nEligibleT4Modules_view_d);
  alpaka::memcpy(queue_, nTotalQuadruplets_buf, nTotalQuadruplets_view_d);
  alpaka::wait(queue_);  // wait for the values before using them

  auto nEligibleT4Modules = *nEligibleT4Modules_buf.data();
  auto nTotalQuadruplets = *nTotalQuadruplets_buf.data();

  if (!quadrupletsDC_) {
    quadrupletsDC_.emplace(queue_, static_cast<int32_t>(nTotalQuadruplets), static_cast<int32_t>(nLowerModules_));
    auto quadrupletsOccupancy = quadrupletsDC_->view().quadrupletsOccupancy();
    auto nQuadruplets_view = cms::alpakatools::make_device_view(
        queue_, quadrupletsOccupancy.nQuadruplets(), quadrupletsOccupancy.metadata().size());
    alpaka::memset(queue_, nQuadruplets_view, 0u);
    auto totOccupancyQuadruplets_view = cms::alpakatools::make_device_view(
        queue_, quadrupletsOccupancy.totOccupancyQuadruplets(), quadrupletsOccupancy.metadata().size());
    alpaka::memset(queue_, totOccupancyQuadruplets_view, 0u);
    auto quadruplets = quadrupletsDC_->view().quadruplets();
    auto isDup_view = cms::alpakatools::make_device_view(queue_, quadruplets.isDup(), quadruplets.metadata().size());
    alpaka::memset(queue_, isDup_view, 0u);
  }

  auto const createQuadruplets_workDiv =
      cms::alpakatools::make_workdiv<Acc3D>({std::max((int)nEligibleT4Modules, 1), 1, 1}, {1, 8, 32});

  alpaka::exec<Acc3D>(queue_,
                      createQuadruplets_workDiv,
                      CreateQuadruplets{},
                      modules_.const_view().modules(),
                      miniDoubletsDC_->const_view().miniDoublets(),
                      segmentsDC_->const_view().segments(),
                      tripletsDC_->view().triplets(),
                      tripletsDC_->const_view().tripletsOccupancy(),
                      quadrupletsDC_->view().quadruplets(),
                      quadrupletsDC_->view().quadrupletsOccupancy(),
                      rangesDC_->const_view(),
                      nEligibleT4Modules,
                      ptCut_);

  auto const removeDupQuadrupletsAfterBuild_workDiv =
      cms::alpakatools::make_workdiv<Acc3D>({max_blocks, 1, 1}, {1, 16, 16});

  alpaka::exec<Acc3D>(queue_,
                      removeDupQuadrupletsAfterBuild_workDiv,
                      RemoveDupQuadrupletsAfterBuild{},
                      modules_.const_view().modules(),
                      quadrupletsDC_->view().quadruplets(),
                      quadrupletsDC_->const_view().quadrupletsOccupancy(),
                      rangesDC_->const_view());

  auto const addQuadrupletRangesToEventExplicit_workDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 1024);

  alpaka::exec<Acc1D>(queue_,
                      addQuadrupletRangesToEventExplicit_workDiv,
                      AddQuadrupletRangesToEventExplicit{},
                      modules_.const_view().modules(),
                      quadrupletsDC_->const_view().quadrupletsOccupancy(),
                      rangesDC_->view());

  if (addObjects_) {
    addQuadrupletsToEventExplicit();
  }
}

void LSTEvent::addMiniDoubletsToEventExplicit() {
  auto nMDsCPU_buf = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, nLowerModules_);
  auto mdsOccupancy = miniDoubletsDC_->const_view().miniDoubletsOccupancy();
  auto nMDs_view =
      cms::alpakatools::make_device_view(queue_, mdsOccupancy.nMDs(), nLowerModules_);  // exclude pixel part
  alpaka::memcpy(queue_, nMDsCPU_buf, nMDs_view, nLowerModules_);

  auto modules = modules_.const_view().modules();

  // FIXME: replace by ES host data
  auto module_subdets_buf = cms::alpakatools::make_host_buffer<short[]>(queue_, nLowerModules_);
  auto module_subdets_view =
      cms::alpakatools::make_device_view(queue_, modules.subdets(), nLowerModules_);  // only lower modules
  alpaka::memcpy(queue_, module_subdets_buf, module_subdets_view, nLowerModules_);

  auto module_layers_buf = cms::alpakatools::make_host_buffer<short[]>(queue_, nLowerModules_);
  auto module_layers_view =
      cms::alpakatools::make_device_view(queue_, modules.layers(), nLowerModules_);  // only lower modules
  alpaka::memcpy(queue_, module_layers_buf, module_layers_view, nLowerModules_);

  alpaka::wait(queue_);  // wait for inputs before using them

  auto const* nMDsCPU = nMDsCPU_buf.data();
  auto const* module_subdets = module_subdets_buf.data();
  auto const* module_layers = module_layers_buf.data();

  for (unsigned int i = 0; i < nLowerModules_; i++) {
    if (nMDsCPU[i] != 0) {
      if (module_subdets[i] == Barrel) {
        n_minidoublets_by_layer_barrel_[module_layers[i] - 1] += nMDsCPU[i];
      } else {
        n_minidoublets_by_layer_endcap_[module_layers[i] - 1] += nMDsCPU[i];
      }
    }
  }
}

void LSTEvent::addSegmentsToEventExplicit() {
  auto nSegmentsCPU_buf = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, nLowerModules_);
  auto nSegments_buf = cms::alpakatools::make_device_view(
      queue_, segmentsDC_->const_view().segmentsOccupancy().nSegments(), nLowerModules_);
  alpaka::memcpy(queue_, nSegmentsCPU_buf, nSegments_buf, nLowerModules_);

  auto modules = modules_.const_view().modules();

  // FIXME: replace by ES host data
  auto module_subdets_buf = cms::alpakatools::make_host_buffer<short[]>(queue_, nLowerModules_);
  auto module_subdets_view =
      cms::alpakatools::make_device_view(queue_, modules.subdets(), nLowerModules_);  // only lower modules
  alpaka::memcpy(queue_, module_subdets_buf, module_subdets_view, nLowerModules_);

  auto module_layers_buf = cms::alpakatools::make_host_buffer<short[]>(queue_, nLowerModules_);
  auto module_layers_view =
      cms::alpakatools::make_device_view(queue_, modules.layers(), nLowerModules_);  // only lower modules
  alpaka::memcpy(queue_, module_layers_buf, module_layers_view, nLowerModules_);

  alpaka::wait(queue_);  // wait for inputs before using them

  auto const* nSegmentsCPU = nSegmentsCPU_buf.data();
  auto const* module_subdets = module_subdets_buf.data();
  auto const* module_layers = module_layers_buf.data();

  for (unsigned int i = 0; i < nLowerModules_; i++) {
    if (!(nSegmentsCPU[i] == 0)) {
      if (module_subdets[i] == Barrel) {
        n_segments_by_layer_barrel_[module_layers[i] - 1] += nSegmentsCPU[i];
      } else {
        n_segments_by_layer_endcap_[module_layers[i] - 1] += nSegmentsCPU[i];
      }
    }
  }
}

void LSTEvent::addQuintupletsToEventExplicit() {
  auto quintupletsOccupancy = quintupletsDC_->const_view().quintupletsOccupancy();
  auto nQuintuplets_view =
      cms::alpakatools::make_device_view(queue_, quintupletsOccupancy.nQuintuplets(), nLowerModules_);
  auto nQuintupletsCPU_buf = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, nLowerModules_);
  alpaka::memcpy(queue_, nQuintupletsCPU_buf, nQuintuplets_view);

  auto modules = modules_.const_view().modules();

  // FIXME: replace by ES host data
  auto module_subdets_buf = cms::alpakatools::make_host_buffer<short[]>(queue_, nLowerModules_);
  auto module_subdets_view =
      cms::alpakatools::make_device_view(queue_, modules.subdets(), nLowerModules_);  // only lower modules
  alpaka::memcpy(queue_, module_subdets_buf, module_subdets_view, nLowerModules_);

  auto module_layers_buf = cms::alpakatools::make_host_buffer<short[]>(queue_, nLowerModules_);
  auto module_layers_view =
      cms::alpakatools::make_device_view(queue_, modules.layers(), nLowerModules_);  // only lower modules
  alpaka::memcpy(queue_, module_layers_buf, module_layers_view, nLowerModules_);

  auto module_quintupletModuleIndices_buf = cms::alpakatools::make_host_buffer<int[]>(queue_, nLowerModules_);
  auto rangesOccupancy = rangesDC_->view();
  auto quintupletModuleIndices_view_d =
      cms::alpakatools::make_device_view(queue_, rangesOccupancy.quintupletModuleIndices(), nLowerModules_);
  alpaka::memcpy(queue_, module_quintupletModuleIndices_buf, quintupletModuleIndices_view_d);

  alpaka::wait(queue_);  // wait for inputs before using them

  auto const* nQuintupletsCPU = nQuintupletsCPU_buf.data();
  auto const* module_subdets = module_subdets_buf.data();
  auto const* module_layers = module_layers_buf.data();
  auto const* module_quintupletModuleIndices = module_quintupletModuleIndices_buf.data();

  for (uint16_t i = 0; i < nLowerModules_; i++) {
    if (!(nQuintupletsCPU[i] == 0 or module_quintupletModuleIndices[i] == -1)) {
      if (module_subdets[i] == Barrel) {
        n_quintuplets_by_layer_barrel_[module_layers[i] - 1] += nQuintupletsCPU[i];
      } else {
        n_quintuplets_by_layer_endcap_[module_layers[i] - 1] += nQuintupletsCPU[i];
      }
    }
  }
}

void LSTEvent::addTripletsToEventExplicit() {
  auto tripletsOccupancy = tripletsDC_->const_view().tripletsOccupancy();
  auto nTriplets_view = cms::alpakatools::make_device_view(queue_, tripletsOccupancy.nTriplets(), nLowerModules_);
  auto nTripletsCPU_buf = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, nLowerModules_);
  alpaka::memcpy(queue_, nTripletsCPU_buf, nTriplets_view);

  auto modules = modules_.const_view().modules();

  // FIXME: replace by ES host data
  auto module_subdets_buf = cms::alpakatools::make_host_buffer<short[]>(queue_, nLowerModules_);
  auto module_subdets_view =
      cms::alpakatools::make_device_view(queue_, modules.subdets(), nLowerModules_);  // only lower modules
  alpaka::memcpy(queue_, module_subdets_buf, module_subdets_view, nLowerModules_);

  auto module_layers_buf = cms::alpakatools::make_host_buffer<short[]>(queue_, nLowerModules_);
  auto module_layers_view =
      cms::alpakatools::make_device_view(queue_, modules.layers(), nLowerModules_);  // only lower modules
  alpaka::memcpy(queue_, module_layers_buf, module_layers_view, nLowerModules_);

  alpaka::wait(queue_);  // wait for inputs before using them

  auto const* nTripletsCPU = nTripletsCPU_buf.data();
  auto const* module_subdets = module_subdets_buf.data();
  auto const* module_layers = module_layers_buf.data();

  for (uint16_t i = 0; i < nLowerModules_; i++) {
    if (nTripletsCPU[i] != 0) {
      if (module_subdets[i] == Barrel) {
        n_triplets_by_layer_barrel_[module_layers[i] - 1] += nTripletsCPU[i];
      } else {
        n_triplets_by_layer_endcap_[module_layers[i] - 1] += nTripletsCPU[i];
      }
    }
  }
}

void LSTEvent::addQuadrupletsToEventExplicit() {
  auto quadrupletsOccupancy = quadrupletsDC_->const_view().quadrupletsOccupancy();
  auto nQuadruplets_view =
      cms::alpakatools::make_device_view(queue_, quadrupletsOccupancy.nQuadruplets(), nLowerModules_);
  auto nQuadrupletsCPU_buf = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, nLowerModules_);
  alpaka::memcpy(queue_, nQuadrupletsCPU_buf, nQuadruplets_view);

  auto modules = modules_.const_view().modules();

  // FIXME: replace by ES host data
  auto module_subdets_buf = cms::alpakatools::make_host_buffer<short[]>(queue_, nLowerModules_);
  auto module_subdets_view =
      cms::alpakatools::make_device_view(queue_, modules.subdets(), nLowerModules_);  // only lower modules
  alpaka::memcpy(queue_, module_subdets_buf, module_subdets_view, nLowerModules_);

  auto module_layers_buf = cms::alpakatools::make_host_buffer<short[]>(queue_, nLowerModules_);
  auto module_layers_view =
      cms::alpakatools::make_device_view(queue_, modules.layers(), nLowerModules_);  // only lower modules
  alpaka::memcpy(queue_, module_layers_buf, module_layers_view, nLowerModules_);

  alpaka::wait(queue_);  // wait for inputs before using them

  auto const* nQuadrupletsCPU = nQuadrupletsCPU_buf.data();
  auto const* module_subdets = module_subdets_buf.data();
  auto const* module_layers = module_layers_buf.data();

  for (uint16_t i = 0; i < nLowerModules_; i++) {
    if (nQuadrupletsCPU[i] != 0) {
      if (module_subdets[i] == Barrel) {
        n_quadruplets_by_layer_barrel_[module_layers[i] - 1] += nQuadrupletsCPU[i];
      } else {
        n_quadruplets_by_layer_endcap_[module_layers[i] - 1] += nQuadrupletsCPU[i];
      }
    }
  }
}

unsigned int LSTEvent::getNumberOfMiniDoublets() {
  unsigned int miniDoublets = 0;
  for (auto& it : n_minidoublets_by_layer_barrel_) {
    miniDoublets += it;
  }
  for (auto& it : n_minidoublets_by_layer_endcap_) {
    miniDoublets += it;
  }

  return miniDoublets;
}

unsigned int LSTEvent::getNumberOfMiniDoubletsByLayerBarrel(unsigned int layer) {
  return n_minidoublets_by_layer_barrel_[layer];
}

unsigned int LSTEvent::getNumberOfMiniDoubletsByLayerEndcap(unsigned int layer) {
  return n_minidoublets_by_layer_endcap_[layer];
}

unsigned int LSTEvent::getNumberOfSegments() {
  unsigned int segments = 0;
  for (auto& it : n_segments_by_layer_barrel_) {
    segments += it;
  }
  for (auto& it : n_segments_by_layer_endcap_) {
    segments += it;
  }

  return segments;
}

unsigned int LSTEvent::getNumberOfSegmentsByLayerBarrel(unsigned int layer) {
  return n_segments_by_layer_barrel_[layer];
}

unsigned int LSTEvent::getNumberOfSegmentsByLayerEndcap(unsigned int layer) {
  return n_segments_by_layer_endcap_[layer];
}

unsigned int LSTEvent::getNumberOfTriplets() {
  unsigned int triplets = 0;
  for (auto& it : n_triplets_by_layer_barrel_) {
    triplets += it;
  }
  for (auto& it : n_triplets_by_layer_endcap_) {
    triplets += it;
  }

  return triplets;
}

unsigned int LSTEvent::getNumberOfTripletsByLayerBarrel(unsigned int layer) {
  return n_triplets_by_layer_barrel_[layer];
}

unsigned int LSTEvent::getNumberOfTripletsByLayerEndcap(unsigned int layer) {
  return n_triplets_by_layer_endcap_[layer];
}

int LSTEvent::getNumberOfPixelTriplets() {
  auto nPixelTriplets_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);

  alpaka::memcpy(
      queue_, nPixelTriplets_buf_h, cms::alpakatools::make_device_view(queue_, (*pixelTripletsDC_)->nPixelTriplets()));
  alpaka::wait(queue_);

  return *nPixelTriplets_buf_h.data();
}

int LSTEvent::getNumberOfPixelQuintuplets() {
  auto nPixelQuintuplets_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);

  alpaka::memcpy(queue_,
                 nPixelQuintuplets_buf_h,
                 cms::alpakatools::make_device_view(queue_, (*pixelQuintupletsDC_)->nPixelQuintuplets()));
  alpaka::wait(queue_);

  return *nPixelQuintuplets_buf_h.data();
}

unsigned int LSTEvent::getNumberOfQuintuplets() {
  unsigned int quintuplets = 0;
  for (auto& it : n_quintuplets_by_layer_barrel_) {
    quintuplets += it;
  }
  for (auto& it : n_quintuplets_by_layer_endcap_) {
    quintuplets += it;
  }

  return quintuplets;
}

unsigned int LSTEvent::getNumberOfQuintupletsByLayerBarrel(unsigned int layer) {
  return n_quintuplets_by_layer_barrel_[layer];
}

unsigned int LSTEvent::getNumberOfQuintupletsByLayerEndcap(unsigned int layer) {
  return n_quintuplets_by_layer_endcap_[layer];
}

int LSTEvent::getNumberOfTrackCandidates() {
  auto nTrackCandidates_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);

  alpaka::memcpy(queue_,
                 nTrackCandidates_buf_h,
                 cms::alpakatools::make_device_view(queue_, (*trackCandidatesBaseDC_)->nTrackCandidates()));
  alpaka::wait(queue_);

  return *nTrackCandidates_buf_h.data();
}

int LSTEvent::getNumberOfPT5TrackCandidates() {
  auto nTrackCandidatesPT5_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);

  alpaka::memcpy(queue_,
                 nTrackCandidatesPT5_buf_h,
                 cms::alpakatools::make_device_view(queue_, (*trackCandidatesExtendedDC_)->nTrackCandidatespT5()));
  alpaka::wait(queue_);

  return *nTrackCandidatesPT5_buf_h.data();
}

int LSTEvent::getNumberOfPT3TrackCandidates() {
  auto nTrackCandidatesPT3_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);

  alpaka::memcpy(queue_,
                 nTrackCandidatesPT3_buf_h,
                 cms::alpakatools::make_device_view(queue_, (*trackCandidatesExtendedDC_)->nTrackCandidatespT3()));
  alpaka::wait(queue_);

  return *nTrackCandidatesPT3_buf_h.data();
}

int LSTEvent::getNumberOfPLSTrackCandidates() {
  auto nTrackCandidatesPLS_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);

  alpaka::memcpy(queue_,
                 nTrackCandidatesPLS_buf_h,
                 cms::alpakatools::make_device_view(queue_, (*trackCandidatesExtendedDC_)->nTrackCandidatespLS()));
  alpaka::wait(queue_);

  return *nTrackCandidatesPLS_buf_h.data();
}

int LSTEvent::getNumberOfPixelTrackCandidates() {
  auto nTrackCandidates_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);
  auto nTrackCandidatesT5_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);
  auto nTrackCandidatesT4_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);

  alpaka::memcpy(queue_,
                 nTrackCandidates_buf_h,
                 cms::alpakatools::make_device_view(queue_, (*trackCandidatesBaseDC_)->nTrackCandidates()));
  alpaka::memcpy(queue_,
                 nTrackCandidatesT5_buf_h,
                 cms::alpakatools::make_device_view(queue_, (*trackCandidatesExtendedDC_)->nTrackCandidatesT5()));
  alpaka::memcpy(queue_,
                 nTrackCandidatesT4_buf_h,
                 cms::alpakatools::make_device_view(queue_, (*trackCandidatesExtendedDC_)->nTrackCandidatesT4()));
  alpaka::wait(queue_);

  return (*nTrackCandidates_buf_h.data()) - (*nTrackCandidatesT5_buf_h.data()) - (*nTrackCandidatesT4_buf_h.data());
}

int LSTEvent::getNumberOfT5TrackCandidates() {
  auto nTrackCandidatesT5_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);

  alpaka::memcpy(queue_,
                 nTrackCandidatesT5_buf_h,
                 cms::alpakatools::make_device_view(queue_, (*trackCandidatesExtendedDC_)->nTrackCandidatesT5()));
  alpaka::wait(queue_);

  return *nTrackCandidatesT5_buf_h.data();
}

int LSTEvent::getNumberOfT4TrackCandidates() {
  auto nTrackCandidatesT4_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);

  alpaka::memcpy(queue_,
                 nTrackCandidatesT4_buf_h,
                 cms::alpakatools::make_device_view(queue_, (*trackCandidatesExtendedDC_)->nTrackCandidatesT4()));
  alpaka::wait(queue_);

  return *nTrackCandidatesT4_buf_h.data();
}

unsigned int LSTEvent::getNumberOfQuadruplets() {
  unsigned int quadruplets = 0;
  for (auto& it : n_quadruplets_by_layer_barrel_) {
    quadruplets += it;
  }
  for (auto& it : n_quadruplets_by_layer_endcap_) {
    quadruplets += it;
  }

  return quadruplets;
}

unsigned int LSTEvent::getNumberOfQuadrupletsByLayerBarrel(unsigned int layer) {
  return n_quadruplets_by_layer_barrel_[layer];
}

unsigned int LSTEvent::getNumberOfQuadrupletsByLayerEndcap(unsigned int layer) {
  return n_quadruplets_by_layer_endcap_[layer];
}

template <typename TSoA, typename TDev>
typename TSoA::ConstView LSTEvent::getInput(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return LSTInputViewAccessor<TSoA>::get(lstInputDC_->const_view());
  } else {
    // In case getTrimmedInput was called first
    if (!lstInputHC_ || lstInputHC_->size()[1] == 0) {
      lstInputHC_.emplace(
          cms::alpakatools::CopyToHost<PortableDeviceCollection<LSTInputSoA, TDev>>::copyAsync(queue_, *lstInputDC_));
      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
    return LSTInputViewAccessor<TSoA>::get(lstInputDC_->const_view());
  }
}
template HitsBaseConst LSTEvent::getInput<HitsBaseSoA>(bool);
template PixelSeedsConst LSTEvent::getInput<PixelSeedsSoA>(bool);

template <typename TSoA, typename TDev>
typename TSoA::ConstView LSTEvent::getHits(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return HitsViewAccessor<TSoA>::get(hitsHC_->const_view());
  } else {
    if (!hitsHC_) {
      hitsHC_.emplace(
          cms::alpakatools::CopyToHost<PortableDeviceCollection<HitsSoA, TDev>>::copyAsync(queue_, *hitsDC_));
      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
    return HitsViewAccessor<TSoA>::get(hitsHC_->const_view());
  }
}
template HitsExtendedConst LSTEvent::getHits<HitsExtendedSoA>(bool);
template HitsRangesConst LSTEvent::getHits<HitsRangesSoA>(bool);

template <typename TDev>
ObjectRangesConst LSTEvent::getRanges(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return rangesDC_->const_view();
  } else {
    if (!rangesHC_) {
      rangesHC_.emplace(
          cms::alpakatools::CopyToHost<PortableDeviceCollection<ObjectRangesSoA, TDev>>::copyAsync(queue_, *rangesDC_));
      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
    return rangesHC_->const_view();
  }
}
template ObjectRangesConst LSTEvent::getRanges<>(bool);

template <typename TSoA, typename TDev>
typename TSoA::ConstView LSTEvent::getMiniDoublets(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return MiniDoubletsViewAccessor<TSoA>::get(miniDoubletsDC_->const_view());
  } else {
    if (!miniDoubletsHC_) {
      miniDoubletsHC_.emplace(
          cms::alpakatools::CopyToHost<PortableDeviceCollection<MiniDoubletsSoABlocks, TDev>>::copyAsync(
              queue_, *miniDoubletsDC_));
      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
    return MiniDoubletsViewAccessor<TSoA>::get(miniDoubletsHC_->const_view());
  }
}
template MiniDoubletsConst LSTEvent::getMiniDoublets<MiniDoubletsSoA>(bool);
template MiniDoubletsOccupancyConst LSTEvent::getMiniDoublets<MiniDoubletsOccupancySoA>(bool);

template <typename TSoA, typename TDev>
typename TSoA::ConstView LSTEvent::getSegments(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return SegmentsViewAccessor<TSoA>::get(segmentsDC_->const_view());
  } else {
    if (!segmentsHC_) {
      segmentsHC_.emplace(cms::alpakatools::CopyToHost<PortableDeviceCollection<SegmentsSoABlocks, TDev>>::copyAsync(
          queue_, *segmentsDC_));
      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
    return SegmentsViewAccessor<TSoA>::get(segmentsHC_->const_view());
  }
}
template SegmentsConst LSTEvent::getSegments<SegmentsSoA>(bool);
template SegmentsOccupancyConst LSTEvent::getSegments<SegmentsOccupancySoA>(bool);

template <typename TDev>
PixelSegmentsConst LSTEvent::getPixelSegments(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return pixelSegmentsDC_->const_view();
  } else {
    if (!pixelSegmentsHC_) {
      pixelSegmentsHC_.emplace(cms::alpakatools::CopyToHost<::PortableCollection<PixelSegmentsSoA, TDev>>::copyAsync(
          queue_, *pixelSegmentsDC_));

      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
  }
  return pixelSegmentsHC_->const_view();
}
template PixelSegmentsConst LSTEvent::getPixelSegments<>(bool);

template <typename TSoA, typename TDev>
typename TSoA::ConstView LSTEvent::getTriplets(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return TripletsViewAccessor<TSoA>::get(tripletsDC_->const_view());
  } else {
    if (!tripletsHC_) {
      tripletsHC_.emplace(cms::alpakatools::CopyToHost<PortableDeviceCollection<TripletsSoABlocks, TDev>>::copyAsync(
          queue_, *tripletsDC_));
      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
  }
  return TripletsViewAccessor<TSoA>::get(tripletsHC_->const_view());
}
template TripletsConst LSTEvent::getTriplets<TripletsSoA>(bool);
template TripletsOccupancyConst LSTEvent::getTriplets<TripletsOccupancySoA>(bool);

template <typename TSoA, typename TDev>
typename TSoA::ConstView LSTEvent::getQuadruplets(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return QuadrupletsViewAccessor<TSoA>::get(quadrupletsDC_->const_view());
  } else {
    if (!quadrupletsHC_) {
      quadrupletsHC_.emplace(
          cms::alpakatools::CopyToHost<PortableDeviceCollection<QuadrupletsSoABlocks, TDev>>::copyAsync(
              queue_, *quadrupletsDC_));
      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
  }
  return QuadrupletsViewAccessor<TSoA>::get(quadrupletsHC_->const_view());
}
template QuadrupletsConst LSTEvent::getQuadruplets<QuadrupletsSoA>(bool);
template QuadrupletsOccupancyConst LSTEvent::getQuadruplets<QuadrupletsOccupancySoA>(bool);

template <typename TSoA, typename TDev>
typename TSoA::ConstView LSTEvent::getQuintuplets(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return QuintupletsViewAccessor<TSoA>::get(quintupletsDC_->const_view());
  } else {
    if (!quintupletsHC_) {
      quintupletsHC_.emplace(
          cms::alpakatools::CopyToHost<PortableDeviceCollection<QuintupletsSoABlocks, TDev>>::copyAsync(
              queue_, *quintupletsDC_));
      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
  }
  return QuintupletsViewAccessor<TSoA>::get(quintupletsHC_->const_view());
}
template QuintupletsConst LSTEvent::getQuintuplets<QuintupletsSoA>(bool);
template QuintupletsOccupancyConst LSTEvent::getQuintuplets<QuintupletsOccupancySoA>(bool);

template <typename TDev>
PixelTripletsConst LSTEvent::getPixelTriplets(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return pixelTripletsDC_->const_view();
  } else {
    if (!pixelTripletsHC_) {
      pixelTripletsHC_.emplace(cms::alpakatools::CopyToHost<::PortableCollection<PixelTripletsSoA, TDev>>::copyAsync(
          queue_, *pixelTripletsDC_));

      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
  }
  return pixelTripletsHC_->const_view();
}
template PixelTripletsConst LSTEvent::getPixelTriplets<>(bool);

template <typename TDev>
PixelQuintupletsConst LSTEvent::getPixelQuintuplets(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return pixelQuintupletsDC_->const_view();
  } else {
    if (!pixelQuintupletsHC_) {
      pixelQuintupletsHC_.emplace(
          cms::alpakatools::CopyToHost<::PortableCollection<PixelQuintupletsSoA, TDev>>::copyAsync(
              queue_, *pixelQuintupletsDC_));

      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
  }
  return pixelQuintupletsHC_->const_view();
}
template PixelQuintupletsConst LSTEvent::getPixelQuintuplets<>(bool);

template <typename TDev>
TrackCandidatesBaseConst LSTEvent::getTrackCandidatesBase(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return trackCandidatesBaseDC_->const_view();
  } else {
    if (!trackCandidatesBaseHC_) {
      trackCandidatesBaseHC_.emplace(
          cms::alpakatools::CopyToHost<::PortableCollection<TrackCandidatesBaseSoA, TDev>>::copyAsync(
              queue_, *trackCandidatesBaseDC_));

      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
  }
  return trackCandidatesBaseHC_->const_view();
}
template TrackCandidatesBaseConst LSTEvent::getTrackCandidatesBase<>(bool);

template <typename TDev>
TrackCandidatesExtendedConst LSTEvent::getTrackCandidatesExtended(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return trackCandidatesExtendedDC_->const_view();
  } else {
    if (!trackCandidatesExtendedHC_) {
      trackCandidatesExtendedHC_.emplace(
          cms::alpakatools::CopyToHost<::PortableCollection<TrackCandidatesExtendedSoA, TDev>>::copyAsync(
              queue_, *trackCandidatesExtendedDC_));

      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
  }
  return trackCandidatesExtendedHC_->const_view();
}
template TrackCandidatesExtendedConst LSTEvent::getTrackCandidatesExtended<>(bool);

std::unique_ptr<TrackCandidatesBaseDeviceCollection> LSTEvent::releaseTrackCandidatesBaseDeviceCollection() {
  return std::make_unique<TrackCandidatesBaseDeviceCollection>(std::move(trackCandidatesBaseDC_.value()));
}

template <typename TSoA, typename TDev>
typename TSoA::ConstView LSTEvent::getModules(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return ModulesViewAccessor<TSoA>::get(modules_.const_view());
  } else {
    if (!modulesHC_) {
      modulesHC_.emplace(
          cms::alpakatools::CopyToHost<PortableDeviceCollection<ModulesSoABlocks, TDev>>::copyAsync(queue_, modules_));
      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
    return ModulesViewAccessor<TSoA>::get(modulesHC_->const_view());
  }
}
template ModulesConst LSTEvent::getModules<ModulesSoA>(bool);
template ModulesPixelConst LSTEvent::getModules<ModulesPixelSoA>(bool);
