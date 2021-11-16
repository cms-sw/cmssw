#include "RecoPixelVertexing/PixelTriplets/plugins/CAHitNtupletGeneratorKernelsImpl.h"

/// Compute the number of quadruplet blocks for block size
inline uint32_t nQuadrupletBlocks(uint32_t blockSize) {
  // caConstants::maxNumberOfQuadruplets is a constexpr, so the compiler will pre compute the 3*max/4
  return (3 * caConstants::maxNumberOfQuadruplets / 4 + blockSize - 1) / blockSize;
}

#define CMS_USE_COOP_GROUPS

#ifndef CMS_USE_COOP_GROUPS
__inline__ void populateMultiplicity(HitContainer const *__restrict__ tuples_d,
                                     Quality const *__restrict__ quality_d,
                                     caConstants::TupleMultiplicity *tupleMultiplicity_d,
                                     cudaStream_t cudaStream) {
  cms::cuda::launchZero(tupleMultiplicity_d, cudaStream);
  auto blockSize = 128;
  auto numberOfBlocks = (3 * caConstants::maxTuples / 4 + blockSize - 1) / blockSize;
  kernel_countOrFillMultiplicity<CountOrFill::count>
      <<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples_d, quality_d, tupleMultiplicity_d);
  cms::cuda::launchFinalize(tupleMultiplicity_d, cudaStream);
  kernel_countOrFillMultiplicity<CountOrFill::fill>
      <<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples_d, quality_d, tupleMultiplicity_d);
}

__inline__ void populateHitInTracks(HitContainer const *__restrict__ tuples_d,
                                    Quality const *__restrict__ quality_d,
                                    CAHitNtupletGeneratorKernelsGPU::HitToTuple::View hitToTupleView,
                                    cudaStream_t cudaStream) {
  auto hitToTuple_d = static_cast<CAHitNtupletGeneratorKernelsGPU::HitToTuple *>(hitToTupleView.assoc);
  cms::cuda::launchZero(hitToTupleView, cudaStream);
  auto blockSize = 64;
  auto numberOfBlocks = nQuadrupletBlocks(blockSize);
  kernel_countOrFillHitInTracks<CountOrFill::count>
      <<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples_d, quality_d, hitToTuple_d);
  cudaCheck(cudaGetLastError());
  cms::cuda::launchFinalize(hitToTupleView, cudaStream);
  cudaCheck(cudaGetLastError());
  kernel__countOrFillHitInTracks<CountOrFill::fill>
      <<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples_d, quality_d, hitToTuple_d);
}

#else
__global__ void kernel_populateHitInTracks(HitContainer const *__restrict__ tuples_d,
                                           Quality const *__restrict__ quality_d,
                                           HitToTuple::View view,
                                           HitToTuple::View::Counter *ws) {
  namespace cg = cooperative_groups;
  auto grid = cg::this_grid();
  auto tuple_d = static_cast<HitToTuple *>(view.assoc);
  zeroAndInitCoop(view);
  grid.sync();
  countOrFillHitInTracks<CountOrFill::count>(tuples_d, quality_d, tuple_d);
  grid.sync();
  finalizeCoop(view, ws);
  grid.sync();
  countOrFillHitInTracks<CountOrFill::fill>(tuples_d, quality_d, tuple_d);
}

__inline__ void populateHitInTracks(HitContainer const *tuples_d,
                                    Quality const *quality_d,
                                    HitToTuple::View view,
                                    cudaStream_t cudaStream) {
  using View = HitToTuple::View;
  int blockSize = 64;
  int nblocks = nQuadrupletBlocks(blockSize);

  auto kernel = kernel_populateHitInTracks;

  assert(nblocks > 0);
  auto nOnes = view.size();
  auto nchunks = nOnes / blockSize + 1;
  auto ws = cms::cuda::make_device_unique<View::Counter[]>(nchunks, cudaStream);
  auto wsp = ws.get();
  // FIXME: discuss with FW team: cuda calls are expensive and not needed for each event
  static int const maxBlocks = maxCoopBlocks(kernel, blockSize, 0, 0);
  auto ncoopblocks = std::min(nblocks, maxBlocks);
  assert(ncoopblocks > 0);
  void *kernelArgs[] = {&tuples_d, &quality_d, &view, &wsp};
  dim3 dimBlock(blockSize, 1, 1);
  dim3 dimGrid(ncoopblocks, 1, 1);
  // launch
  cudaCheck(cudaLaunchCooperativeKernel((void *)kernel, dimGrid, dimBlock, kernelArgs, 0, cudaStream));
}

__global__ void kernel_populateMultiplicity(HitContainer const *__restrict__ tuples_d,
                                            Quality const *__restrict__ quality_d,
                                            caConstants::TupleMultiplicity::View view,
                                            caConstants::TupleMultiplicity::View::Counter *ws) {
  namespace cg = cooperative_groups;
  auto grid = cg::this_grid();
  auto tupleMultiplicity_d = static_cast<caConstants::TupleMultiplicity *>(view.assoc);
  zeroAndInitCoop(view);
  grid.sync();
  countOrFillMultiplicity<CountOrFill::count>(tuples_d, quality_d, tupleMultiplicity_d);
  grid.sync();
  finalizeCoop(view, ws);
  grid.sync();
  countOrFillMultiplicity<CountOrFill::fill>(tuples_d, quality_d, tupleMultiplicity_d);
}

__inline__ void populateMultiplicity(HitContainer const *tuples_d,
                                     Quality const *quality_d,
                                     caConstants::TupleMultiplicity *tupleMultiplicity_d,
                                     cudaStream_t cudaStream) {
  auto kernel = kernel_populateMultiplicity;
  using View = caConstants::TupleMultiplicity::View;
  View view = {tupleMultiplicity_d, nullptr, nullptr, -1, -1};

  int blockSize = 128;
  int nblocks = (3 * caConstants::maxTuples / 4 + blockSize - 1) / blockSize;
  assert(nblocks > 0);
  auto nOnes = view.size();
  auto nchunks = nOnes / blockSize + 1;
  auto ws = cms::cuda::make_device_unique<View::Counter[]>(nchunks, cudaStream);
  auto wsp = ws.get();
  // FIXME: discuss with FW team: cuda calls are expensive and not needed for each event
  static int maxBlocks = maxCoopBlocks(kernel, blockSize, 0, 0);
  auto ncoopblocks = std::min(nblocks, maxBlocks);
  assert(ncoopblocks > 0);
  void *kernelArgs[] = {&tuples_d, &quality_d, &view, &wsp};
  dim3 dimBlock(blockSize, 1, 1);
  dim3 dimGrid(ncoopblocks, 1, 1);
  // launch
  cudaCheck(cudaLaunchCooperativeKernel((void *)kernel, dimGrid, dimBlock, kernelArgs, 0, cudaStream));
}

#endif

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
      isOuterHitOfCell_,
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
    auto numberOfBlocks = (nhits - isOuterHitOfCell_.offset + blockSize - 1) / blockSize;
    dim3 blks(1, numberOfBlocks, 1);
    dim3 thrs(stride, blockSize, 1);
    gpuPixelDoublets::fishbone<<<blks, thrs, 0, cudaStream>>>(
        hh.view(), device_theCells_.get(), device_nCells_, isOuterHitOfCell_, nhits, false);
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

  populateMultiplicity(tuples_d, quality_d, device_tupleMultiplicity_.get(), cudaStream);
  cudaCheck(cudaGetLastError());

  if (nhits > 1 && params_.lateFishbone_) {
    auto nthTot = 128;
    auto stride = 16;
    auto blockSize = nthTot / stride;
    auto numberOfBlocks = (nhits - isOuterHitOfCell_.offset + blockSize - 1) / blockSize;
    dim3 blks(1, numberOfBlocks, 1);
    dim3 thrs(stride, blockSize, 1);
    gpuPixelDoublets::fishbone<<<blks, thrs, 0, cudaStream>>>(
        hh.view(), device_theCells_.get(), device_nCells_, isOuterHitOfCell_, nhits, true);
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

  isOuterHitOfCell_ = GPUCACell::OuterHitOfCell{device_isOuterHitOfCell_.get(), hh.offsetBPIX2()};

#ifdef NTUPLE_DEBUG
  std::cout << "building Doublets out of " << nhits << " Hits" << std::endl;
#endif

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

  // in principle we can use "nhits" to heuristically dimension the workspace...
  device_isOuterHitOfCell_ = cms::cuda::make_device_unique<GPUCACell::OuterHitOfCellContainer[]>(
      std::max(1, nhits - hh.offsetBPIX2()), stream);
  assert(device_isOuterHitOfCell_.get());

  isOuterHitOfCell_ = GPUCACell::OuterHitOfCell{device_isOuterHitOfCell_.get(), hh.offsetBPIX2()};

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
    int blocks = (std::max(1, nhits - hh.offsetBPIX2()) + threadsPerBlock - 1) / threadsPerBlock;
    gpuPixelDoublets::initDoublets<<<blocks, threadsPerBlock, 0, stream>>>(isOuterHitOfCell_,
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
                                                                    isOuterHitOfCell_,
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
    // populate hit->track "map"
    populateHitInTracks(tuples_d, quality_d, hitToTupleView_, cudaStream);
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
                                                                        isOuterHitOfCell_,
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
