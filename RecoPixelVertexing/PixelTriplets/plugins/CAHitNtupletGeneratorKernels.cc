#include "RecoPixelVertexing/PixelTriplets/plugins/CAHitNtupletGeneratorKernelsImpl.h"

template <>
void CAHitNtupletGeneratorKernelsCPU::printCounters(Counters const *counters) {
  kernel_printCounters(counters);
}

template <>
void CAHitNtupletGeneratorKernelsCPU::fillHitDetIndices(HitsView const *hv, TkSoA *tracks_d, cudaStream_t) {
  kernel_fillHitDetIndices(&tracks_d->hitIndices, hv, &tracks_d->detIndices);
}

template <>
void CAHitNtupletGeneratorKernelsCPU::buildDoublets(HitsOnCPU const &hh, cudaStream_t stream) {
  auto nhits = hh.nHits();

#ifdef NTUPLE_DEBUG
  std::cout << "building Doublets out of " << nhits << " Hits" << std::endl;
#endif

  // use "nhits" to heuristically dimension the workspace

  // no need to use the Traits allocations, since we know this is being compiled for the CPU
  //device_isOuterHitOfCell_ = Traits::template make_unique<GPUCACell::OuterHitOfCell[]>(std::max(1U, nhits), stream);
  device_isOuterHitOfCell_ = std::make_unique<GPUCACell::OuterHitOfCell[]>(std::max(1U, nhits));
  assert(device_isOuterHitOfCell_.get());

  auto cellStorageSize = caConstants::maxNumOfActiveDoublets * sizeof(GPUCACell::CellNeighbors) +
                         caConstants::maxNumOfActiveDoublets * sizeof(GPUCACell::CellTracks);
  // no need to use the Traits allocations, since we know this is being compiled for the CPU
  //cellStorage_ = Traits::template make_unique<unsigned char[]>(cellStorageSize, stream);
  cellStorage_ = std::make_unique<unsigned char[]>(cellStorageSize);
  device_theCellNeighborsContainer_ = (GPUCACell::CellNeighbors *)cellStorage_.get();
  device_theCellTracksContainer_ = (GPUCACell::CellTracks *)(cellStorage_.get() + caConstants::maxNumOfActiveDoublets *
                                                                                      sizeof(GPUCACell::CellNeighbors));

  gpuPixelDoublets::initDoublets(device_isOuterHitOfCell_.get(),
                                 nhits,
                                 device_theCellNeighbors_.get(),
                                 device_theCellNeighborsContainer_,
                                 device_theCellTracks_.get(),
                                 device_theCellTracksContainer_);

  // no need to use the Traits allocations, since we know this is being compiled for the CPU
  //device_theCells_ = Traits::template make_unique<GPUCACell[]>(params_.maxNumberOfDoublets_, stream);
  device_theCells_ = std::make_unique<GPUCACell[]>(params_.maxNumberOfDoublets_);
  if (0 == nhits)
    return;  // protect against empty events

  // take all layer pairs into account
  auto nActualPairs = gpuPixelDoublets::nPairs;
  if (not params_.includeJumpingForwardDoublets_) {
    // exclude forward "jumping" layer pairs
    nActualPairs = gpuPixelDoublets::nPairsForTriplets;
  }
  if (params_.minHitsPerNtuplet_ > 3) {
    // for quadruplets, exclude all "jumping" layer pairs
    nActualPairs = gpuPixelDoublets::nPairsForQuadruplets;
  }

  assert(nActualPairs <= gpuPixelDoublets::nPairs);
  gpuPixelDoublets::getDoubletsFromHisto(device_theCells_.get(),
                                         device_nCells_,
                                         device_theCellNeighbors_.get(),
                                         device_theCellTracks_.get(),
                                         hh.view(),
                                         device_isOuterHitOfCell_.get(),
                                         nActualPairs,
                                         params_.idealConditions_,
                                         params_.doClusterCut_,
                                         params_.doZ0Cut_,
                                         params_.doPtCut_,
                                         params_.maxNumberOfDoublets_);
}

template <>
void CAHitNtupletGeneratorKernelsCPU::launchKernels(HitsOnCPU const &hh, TkSoA *tracks_d, cudaStream_t cudaStream) {
  auto *tuples_d = &tracks_d->hitIndices;
  auto *quality_d = tracks_d->qualityData();

  assert(tuples_d && quality_d);

  // zero tuples
  cms::cuda::launchZero(tuples_d, cudaStream);

  auto nhits = hh.nHits();

  // std::cout << "N hits " << nhits << std::endl;
  // if (nhits<2) std::cout << "too few hits " << nhits << std::endl;

  //
  // applying conbinatoric cleaning such as fishbone at this stage is too expensive
  //

  kernel_connect(device_hitTuple_apc_,
                 device_hitToTuple_apc_,  // needed only to be reset, ready for next kernel
                 hh.view(),
                 device_theCells_.get(),
                 device_nCells_,
                 device_theCellNeighbors_.get(),
                 device_isOuterHitOfCell_.get(),
                 params_.hardCurvCut_,
                 params_.ptmin_,
                 params_.CAThetaCutBarrel_,
                 params_.CAThetaCutForward_,
                 params_.dcaCutInnerTriplet_,
                 params_.dcaCutOuterTriplet_);

  if (nhits > 1 && params_.earlyFishbone_) {
    gpuPixelDoublets::fishbone(
        hh.view(), device_theCells_.get(), device_nCells_, device_isOuterHitOfCell_.get(), nhits, false);
  }

  kernel_find_ntuplets(hh.view(),
                       device_theCells_.get(),
                       device_nCells_,
                       device_theCellTracks_.get(),
                       tuples_d,
                       device_hitTuple_apc_,
                       quality_d,
                       params_.minHitsPerNtuplet_);
  if (params_.doStats_)
    kernel_mark_used(hh.view(), device_theCells_.get(), device_nCells_);

  cms::cuda::finalizeBulk(device_hitTuple_apc_, tuples_d);

  // remove duplicates (tracks that share a doublet)
  kernel_earlyDuplicateRemover(device_theCells_.get(), device_nCells_, tuples_d, quality_d, params_.dupPassThrough_);

  kernel_countMultiplicity(tuples_d, quality_d, device_tupleMultiplicity_.get());
  cms::cuda::launchFinalize(device_tupleMultiplicity_.get(), cudaStream);
  kernel_fillMultiplicity(tuples_d, quality_d, device_tupleMultiplicity_.get());

  if (nhits > 1 && params_.lateFishbone_) {
    gpuPixelDoublets::fishbone(
        hh.view(), device_theCells_.get(), device_nCells_, device_isOuterHitOfCell_.get(), nhits, true);
  }

  if (params_.doStats_) {
    kernel_checkOverflows(tuples_d,
                          device_tupleMultiplicity_.get(),
                          device_hitToTuple_.get(),
                          device_hitTuple_apc_,
                          device_theCells_.get(),
                          device_nCells_,
                          device_theCellNeighbors_.get(),
                          device_theCellTracks_.get(),
                          device_isOuterHitOfCell_.get(),
                          nhits,
                          params_.maxNumberOfDoublets_,
                          counters_);
  }
}

template <>
void CAHitNtupletGeneratorKernelsCPU::classifyTuples(HitsOnCPU const &hh, TkSoA *tracks_d, cudaStream_t cudaStream) {
  auto const *tuples_d = &tracks_d->hitIndices;
  auto *quality_d = tracks_d->qualityData();

  // classify tracks based on kinematics
  kernel_classifyTracks(tuples_d, tracks_d, params_.cuts_, quality_d);

  if (params_.lateFishbone_) {
    // apply fishbone cleaning to good tracks
    kernel_fishboneCleaner(device_theCells_.get(), device_nCells_, quality_d);
  }

  // remove duplicates (tracks that share a doublet)
  kernel_fastDuplicateRemover(device_theCells_.get(), device_nCells_, tuples_d, tracks_d, params_.dupPassThrough_);

  // fill hit->track "map"
  if (params_.doSharedHitCut_ || params_.doStats_) {
    kernel_countHitInTracks(tuples_d, quality_d, device_hitToTuple_.get());
    cms::cuda::launchFinalize(hitToTupleView_, cudaStream);
    kernel_fillHitInTracks(tuples_d, quality_d, device_hitToTuple_.get());
  }

  // remove duplicates (tracks that share at least one hit)
  if (params_.doSharedHitCut_) {
    kernel_rejectDuplicate(hh.view(),
                           tuples_d,
                           tracks_d,
                           quality_d,
                           params_.minHitsForSharingCut_,
                           params_.dupPassThrough_,
                           device_hitToTuple_.get());

    kernel_sharedHitCleaner(hh.view(),
                            tuples_d,
                            tracks_d,
                            quality_d,
                            params_.minHitsForSharingCut_,
                            params_.dupPassThrough_,
                            device_hitToTuple_.get());
    if (params_.useSimpleTripletCleaner_) {
      kernel_simpleTripletCleaner(hh.view(),
                                  tuples_d,
                                  tracks_d,
                                  quality_d,
                                  params_.minHitsForSharingCut_,
                                  params_.dupPassThrough_,
                                  device_hitToTuple_.get());
    } else {
      kernel_tripletCleaner(hh.view(),
                            tuples_d,
                            tracks_d,
                            quality_d,
                            params_.minHitsForSharingCut_,
                            params_.dupPassThrough_,
                            device_hitToTuple_.get());
    }
  }
  if (params_.doStats_) {
    // counters (add flag???)
    kernel_doStatsForHitInTracks(device_hitToTuple_.get(), counters_);
    kernel_doStatsForTracks(tuples_d, quality_d, counters_);
  }

#ifdef DUMP_GPU_TK_TUPLES
  static std::atomic<int> iev(0);
  ++iev;
  kernel_print_found_ntuplets(hh.view(), tuples_d, tracks_d, quality_d, device_hitToTuple_.get(), 100, iev);
#endif
}
