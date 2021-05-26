#include "RecoPixelVertexing/PixelTriplets/plugins/CAHitNtupletGeneratorKernelsImpl.h"

template <>
void CAHitNtupletGeneratorKernelsGPU::fillHitDetIndices(HitsView const *hv, TkSoA *tracks_d, cudaStream_t cudaStream) {
  auto blockSize = 128;
  auto numberOfBlocks = (HitContainer::ctCapacity() + blockSize - 1) / blockSize;

  kernel_fillHitDetIndices<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
      &tracks_d->hitIndices, hv, &tracks_d->detIndices);
  cudaCheck(cudaGetLastError());
#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif
}

template <>
void CAHitNtupletGeneratorKernelsGPU::launchKernels(HitsOnCPU const &hh, TkSoA *tracks_d, cudaStream_t cudaStream) {
  // these are pointer on GPU!
  auto *tuples_d = &tracks_d->hitIndices;
  auto *quality_d = tracks_d->qualityData();

  // zero tuples
  cms::cuda::launchZero(tuples_d, cudaStream);

  auto nhits = hh.nHits();

#ifdef NTUPLE_DEBUG
  std::cout << "start tuple building. N hits " << nhits << std::endl;
  if (nhits < 2)
    std::cout << "too few hits " << nhits << std::endl;
#endif

  //
  // applying conbinatoric cleaning such as fishbone at this stage is too expensive
  //

  auto nthTot = 64;
  auto stride = 4;
  auto blockSize = nthTot / stride;
  auto numberOfBlocks = nDoubletBlocks(blockSize);
  auto rescale = numberOfBlocks / 65536;
  blockSize *= (rescale + 1);
  numberOfBlocks = nDoubletBlocks(blockSize);
  assert(numberOfBlocks < 65536);
  assert(blockSize > 0 && 0 == blockSize % 16);
  dim3 blks(1, numberOfBlocks, 1);
  dim3 thrs(stride, blockSize, 1);

  kernel_connect<<<blks, thrs, 0, cudaStream>>>(
      device_hitTuple_apc_,
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
  cudaCheck(cudaGetLastError());

  if (nhits > 1 && params_.earlyFishbone_) {
    auto nthTot = 128;
    auto stride = 16;
    auto blockSize = nthTot / stride;
    auto numberOfBlocks = (nhits + blockSize - 1) / blockSize;
    dim3 blks(1, numberOfBlocks, 1);
    dim3 thrs(stride, blockSize, 1);
    gpuPixelDoublets::fishbone<<<blks, thrs, 0, cudaStream>>>(
        hh.view(), device_theCells_.get(), device_nCells_, device_isOuterHitOfCell_.get(), nhits, false);
    cudaCheck(cudaGetLastError());
  }

  blockSize = 64;
  numberOfBlocks = (3 * params_.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  kernel_find_ntuplets<<<numberOfBlocks, blockSize, 0, cudaStream>>>(hh.view(),
                                                                     device_theCells_.get(),
                                                                     device_nCells_,
                                                                     device_theCellTracks_.get(),
                                                                     tuples_d,
                                                                     device_hitTuple_apc_,
                                                                     quality_d,
                                                                     params_.minHitsPerNtuplet_);
  cudaCheck(cudaGetLastError());

  if (params_.doStats_)
    kernel_mark_used<<<numberOfBlocks, blockSize, 0, cudaStream>>>(hh.view(), device_theCells_.get(), device_nCells_);
  cudaCheck(cudaGetLastError());

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

  blockSize = 128;
  numberOfBlocks = (HitContainer::ctNOnes() + blockSize - 1) / blockSize;
  cms::cuda::finalizeBulk<<<numberOfBlocks, blockSize, 0, cudaStream>>>(device_hitTuple_apc_, tuples_d);

  // remove duplicates (tracks that share a doublet)
  numberOfBlocks = nDoubletBlocks(blockSize);
  kernel_earlyDuplicateRemover<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
      device_theCells_.get(), device_nCells_, tuples_d, quality_d, params_.dupPassThrough_);
  cudaCheck(cudaGetLastError());

  blockSize = 128;
  numberOfBlocks = (3 * caConstants::maxTuples / 4 + blockSize - 1) / blockSize;
  kernel_countMultiplicity<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
      tuples_d, quality_d, device_tupleMultiplicity_.get());
  cms::cuda::launchFinalize(device_tupleMultiplicity_.get(), cudaStream);
  kernel_fillMultiplicity<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
      tuples_d, quality_d, device_tupleMultiplicity_.get());
  cudaCheck(cudaGetLastError());

  if (nhits > 1 && params_.lateFishbone_) {
    auto nthTot = 128;
    auto stride = 16;
    auto blockSize = nthTot / stride;
    auto numberOfBlocks = (nhits + blockSize - 1) / blockSize;
    dim3 blks(1, numberOfBlocks, 1);
    dim3 thrs(stride, blockSize, 1);
    gpuPixelDoublets::fishbone<<<blks, thrs, 0, cudaStream>>>(
        hh.view(), device_theCells_.get(), device_nCells_, device_isOuterHitOfCell_.get(), nhits, true);
    cudaCheck(cudaGetLastError());
  }

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

  // free space asap
  // device_isOuterHitOfCell_.reset();
}

template <>
void CAHitNtupletGeneratorKernelsGPU::buildDoublets(HitsOnCPU const &hh, cudaStream_t stream) {
  int32_t nhits = hh.nHits();

#ifdef NTUPLE_DEBUG
  std::cout << "building Doublets out of " << nhits << " Hits" << std::endl;
#endif

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

  // in principle we can use "nhits" to heuristically dimension the workspace...
  device_isOuterHitOfCell_ = cms::cuda::make_device_unique<GPUCACell::OuterHitOfCell[]>(std::max(1, nhits), stream);
  assert(device_isOuterHitOfCell_.get());

  cellStorage_ = cms::cuda::make_device_unique<unsigned char[]>(
      caConstants::maxNumOfActiveDoublets * sizeof(GPUCACell::CellNeighbors) +
          caConstants::maxNumOfActiveDoublets * sizeof(GPUCACell::CellTracks),
      stream);
  device_theCellNeighborsContainer_ = (GPUCACell::CellNeighbors *)cellStorage_.get();
  device_theCellTracksContainer_ = (GPUCACell::CellTracks *)(cellStorage_.get() + caConstants::maxNumOfActiveDoublets *
                                                                                      sizeof(GPUCACell::CellNeighbors));

  {
    int threadsPerBlock = 128;
    // at least one block!
    int blocks = (std::max(1, nhits) + threadsPerBlock - 1) / threadsPerBlock;
    gpuPixelDoublets::initDoublets<<<blocks, threadsPerBlock, 0, stream>>>(device_isOuterHitOfCell_.get(),
                                                                           nhits,
                                                                           device_theCellNeighbors_.get(),
                                                                           device_theCellNeighborsContainer_,
                                                                           device_theCellTracks_.get(),
                                                                           device_theCellTracksContainer_);
    cudaCheck(cudaGetLastError());
  }

  device_theCells_ = cms::cuda::make_device_unique<GPUCACell[]>(params_.maxNumberOfDoublets_, stream);

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

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
  int stride = 4;
  int threadsPerBlock = gpuPixelDoublets::getDoubletsFromHistoMaxBlockSize / stride;
  int blocks = (4 * nhits + threadsPerBlock - 1) / threadsPerBlock;
  dim3 blks(1, blocks, 1);
  dim3 thrs(stride, threadsPerBlock, 1);
  gpuPixelDoublets::getDoubletsFromHisto<<<blks, thrs, 0, stream>>>(device_theCells_.get(),
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
  cudaCheck(cudaGetLastError());

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif
}

template <>
void CAHitNtupletGeneratorKernelsGPU::classifyTuples(HitsOnCPU const &hh, TkSoA *tracks_d, cudaStream_t cudaStream) {
  // these are pointer on GPU!
  auto const *tuples_d = &tracks_d->hitIndices;
  auto *quality_d = tracks_d->qualityData();

  int32_t nhits = hh.nHits();

  auto blockSize = 64;

  // classify tracks based on kinematics
  auto numberOfBlocks = nQuadrupletBlocks(blockSize);
  kernel_classifyTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples_d, tracks_d, params_.cuts_, quality_d);
  cudaCheck(cudaGetLastError());

  if (params_.lateFishbone_) {
    // apply fishbone cleaning to good tracks
    numberOfBlocks = nDoubletBlocks(blockSize);
    kernel_fishboneCleaner<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
        device_theCells_.get(), device_nCells_, quality_d);
    cudaCheck(cudaGetLastError());
  }

  // mark duplicates (tracks that share a doublet)
  numberOfBlocks = nDoubletBlocks(blockSize);
  kernel_fastDuplicateRemover<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
      device_theCells_.get(), device_nCells_, tuples_d, tracks_d, params_.dupPassThrough_);
  cudaCheck(cudaGetLastError());
#ifdef GPU_DEBUG
  cudaCheck(cudaDeviceSynchronize());
#endif

  if (params_.doSharedHitCut_ || params_.doStats_) {
    // fill hit->track "map"
    assert(hitToTupleView_.offSize > nhits);
    numberOfBlocks = nQuadrupletBlocks(blockSize);
    kernel_countHitInTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
        tuples_d, quality_d, device_hitToTuple_.get());
    cudaCheck(cudaGetLastError());
    assert((hitToTupleView_.assoc == device_hitToTuple_.get()) &&
           (hitToTupleView_.offStorage == device_hitToTupleStorage_.get()) && (hitToTupleView_.offSize > 0));
    cms::cuda::launchFinalize(hitToTupleView_, cudaStream);
    cudaCheck(cudaGetLastError());
    kernel_fillHitInTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples_d, quality_d, device_hitToTuple_.get());
    cudaCheck(cudaGetLastError());
#ifdef GPU_DEBUG
    cudaCheck(cudaDeviceSynchronize());
#endif
  }

  if (params_.doSharedHitCut_) {
    // mark duplicates (tracks that share at least one hit)
    numberOfBlocks = (hitToTupleView_.offSize + blockSize - 1) / blockSize;

    kernel_rejectDuplicate<<<numberOfBlocks, blockSize, 0, cudaStream>>>(hh.view(),
                                                                         tuples_d,
                                                                         tracks_d,
                                                                         quality_d,
                                                                         params_.minHitsForSharingCut_,
                                                                         params_.dupPassThrough_,
                                                                         device_hitToTuple_.get());

    kernel_sharedHitCleaner<<<numberOfBlocks, blockSize, 0, cudaStream>>>(hh.view(),
                                                                          tuples_d,
                                                                          tracks_d,
                                                                          quality_d,
                                                                          params_.minHitsForSharingCut_,
                                                                          params_.dupPassThrough_,
                                                                          device_hitToTuple_.get());

    if (params_.useSimpleTripletCleaner_) {
      kernel_simpleTripletCleaner<<<numberOfBlocks, blockSize, 0, cudaStream>>>(hh.view(),
                                                                                tuples_d,
                                                                                tracks_d,
                                                                                quality_d,
                                                                                params_.minHitsForSharingCut_,
                                                                                params_.dupPassThrough_,
                                                                                device_hitToTuple_.get());
    } else {
      kernel_tripletCleaner<<<numberOfBlocks, blockSize, 0, cudaStream>>>(hh.view(),
                                                                          tuples_d,
                                                                          tracks_d,
                                                                          quality_d,
                                                                          params_.minHitsForSharingCut_,
                                                                          params_.dupPassThrough_,
                                                                          device_hitToTuple_.get());
    }
    cudaCheck(cudaGetLastError());
#ifdef GPU_DEBUG
    cudaCheck(cudaDeviceSynchronize());
#endif
  }

  if (params_.doStats_) {
    numberOfBlocks = (std::max(nhits, int(params_.maxNumberOfDoublets_)) + blockSize - 1) / blockSize;
    kernel_checkOverflows<<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples_d,
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
    cudaCheck(cudaGetLastError());
  }

  if (params_.doStats_) {
    // counters (add flag???)
    numberOfBlocks = (hitToTupleView_.offSize + blockSize - 1) / blockSize;
    kernel_doStatsForHitInTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(device_hitToTuple_.get(), counters_);
    cudaCheck(cudaGetLastError());
    numberOfBlocks = (3 * caConstants::maxNumberOfQuadruplets / 4 + blockSize - 1) / blockSize;
    kernel_doStatsForTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples_d, quality_d, counters_);
    cudaCheck(cudaGetLastError());
  }
#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

#ifdef DUMP_GPU_TK_TUPLES
  static std::atomic<int> iev(0);
  ++iev;
  kernel_print_found_ntuplets<<<1, 32, 0, cudaStream>>>(
      hh.view(), tuples_d, tracks_d, quality_d, device_hitToTuple_.get(), 100, iev);
#endif
}

template <>
void CAHitNtupletGeneratorKernelsGPU::printCounters(Counters const *counters) {
  kernel_printCounters<<<1, 1>>>(counters);
}
