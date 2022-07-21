#include "RecoPixelVertexing/PixelTriplets/plugins/CAHitNtupletGeneratorKernelsImpl.h"
#include <mutex>

// #define NTUPLE_DEBUG
// #define GPU_DEBUG

template <typename TrackerTraits>
void CAHitNtupletGeneratorKernelsGPU<TrackerTraits>::launchKernels(HitsOnCPU const &hh,
                                                                   TkSoA *tracks_d,
                                                                   cudaStream_t cudaStream) {
  using namespace gpuPixelDoublets;
  using namespace caHitNtupletGeneratorKernels;
  // these are pointer on GPU!
  auto *tuples_d = &tracks_d->hitIndices;
  auto *detId_d = &tracks_d->detIndices;
  auto *quality_d = tracks_d->qualityData();

  // zero tuples
  cms::cuda::launchZero(tuples_d, cudaStream);

  int32_t nhits = hh.nHits();

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
  auto numberOfBlocks = this->nDoubletBlocks(blockSize);
  auto rescale = numberOfBlocks / 65536;
  blockSize *= (rescale + 1);
  numberOfBlocks = this->nDoubletBlocks(blockSize);
  assert(numberOfBlocks < 65536);
  assert(blockSize > 0 && 0 == blockSize % 16);
  dim3 blks(1, numberOfBlocks, 1);
  dim3 thrs(stride, blockSize, 1);

  kernel_connect<TrackerTraits>
      <<<blks, thrs, 0, cudaStream>>>(this->device_hitTuple_apc_,
                                      this->device_hitToTuple_apc_,  // needed only to be reset, ready for next kernel
                                      hh.view(),
                                      this->device_theCells_.get(),
                                      this->device_nCells_,
                                      this->device_theCellNeighbors_.get(),
                                      this->isOuterHitOfCell_,
                                      this->params_.caParams_);

  cudaCheck(cudaGetLastError());

  // do not run the fishbone if there are hits only in BPIX1
  if (nhits > this->isOuterHitOfCell_.offset && this->params_.earlyFishbone_) {
    auto nthTot = 128;
    auto stride = 16;
    auto blockSize = nthTot / stride;
    auto numberOfBlocks = (nhits - this->isOuterHitOfCell_.offset + blockSize - 1) / blockSize;
    dim3 blks(1, numberOfBlocks, 1);
    dim3 thrs(stride, blockSize, 1);
    fishbone<TrackerTraits><<<blks, thrs, 0, cudaStream>>>(
        hh.view(), this->device_theCells_.get(), this->device_nCells_, this->isOuterHitOfCell_, nhits, false);
    cudaCheck(cudaGetLastError());
  }

  blockSize = 64;
  numberOfBlocks = (3 * this->params_.cellCuts_.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  kernel_find_ntuplets<TrackerTraits><<<numberOfBlocks, blockSize, 0, cudaStream>>>(hh.view(),
                                                                                    this->device_theCells_.get(),
                                                                                    this->device_nCells_,
                                                                                    this->device_theCellTracks_.get(),
                                                                                    tuples_d,
                                                                                    this->device_hitTuple_apc_,
                                                                                    quality_d,
                                                                                    this->params_.caParams_);
#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif
  if (this->params_.doStats_)
    kernel_mark_used<TrackerTraits>
        <<<numberOfBlocks, blockSize, 0, cudaStream>>>(this->device_theCells_.get(), this->device_nCells_);
  cudaCheck(cudaGetLastError());

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

  blockSize = 128;
  numberOfBlocks = (HitContainer::ctNOnes() + blockSize - 1) / blockSize;

  cms::cuda::finalizeBulk<<<numberOfBlocks, blockSize, 0, cudaStream>>>(this->device_hitTuple_apc_, tuples_d);

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

  kernel_fillHitDetIndices<TrackerTraits><<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples_d, hh.view(), detId_d);
  cudaCheck(cudaGetLastError());

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif
  kernel_fillNLayers<TrackerTraits><<<numberOfBlocks, blockSize, 0, cudaStream>>>(tracks_d, this->device_hitTuple_apc_);
  cudaCheck(cudaGetLastError());

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

  // remove duplicates (tracks that share a doublet)
  numberOfBlocks = this->nDoubletBlocks(blockSize);

  kernel_earlyDuplicateRemover<TrackerTraits><<<numberOfBlocks, blockSize, 0, cudaStream>>>(
      this->device_theCells_.get(), this->device_nCells_, tracks_d, quality_d, this->params_.dupPassThrough_);
  cudaCheck(cudaGetLastError());
#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

  blockSize = 128;
  numberOfBlocks = (3 * TrackerTraits::maxNumberOfTuples / 4 + blockSize - 1) / blockSize;
  kernel_countMultiplicity<TrackerTraits>
      <<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples_d, quality_d, this->device_tupleMultiplicity_.get());
  cms::cuda::launchFinalize(this->device_tupleMultiplicity_.get(), cudaStream);
  kernel_fillMultiplicity<TrackerTraits>
      <<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples_d, quality_d, this->device_tupleMultiplicity_.get());
  cudaCheck(cudaGetLastError());
#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

  // do not run the fishbone if there are hits only in BPIX1
  if (nhits > this->isOuterHitOfCell_.offset && this->params_.lateFishbone_) {
    auto nthTot = 128;
    auto stride = 16;
    auto blockSize = nthTot / stride;
    auto numberOfBlocks = (nhits - this->isOuterHitOfCell_.offset + blockSize - 1) / blockSize;
    dim3 blks(1, numberOfBlocks, 1);
    dim3 thrs(stride, blockSize, 1);
    fishbone<TrackerTraits><<<blks, thrs, 0, cudaStream>>>(
        hh.view(), this->device_theCells_.get(), this->device_nCells_, this->isOuterHitOfCell_, nhits, true);
    cudaCheck(cudaGetLastError());
  }

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

  // free space asap
  // this->device_isOuterHitOfCell_.reset();
}

template <typename TrackerTraits>
void CAHitNtupletGeneratorKernelsGPU<TrackerTraits>::buildDoublets(HitsOnCPU const &hh, cudaStream_t stream) {
  int32_t nhits = hh.nHits();

  using namespace gpuPixelDoublets;

  using GPUCACell = GPUCACellT<TrackerTraits>;
  using OuterHitOfCell = typename GPUCACell::OuterHitOfCell;
  using CellNeighbors = typename GPUCACell::CellNeighbors;
  using CellTracks = typename GPUCACell::CellTracks;
  using OuterHitOfCellContainer = typename GPUCACell::OuterHitOfCellContainer;

  this->isOuterHitOfCell_ = OuterHitOfCell{this->device_isOuterHitOfCell_.get(), hh.offsetBPIX2()};

#ifdef NTUPLE_DEBUG
  std::cout << "building Doublets out of " << nhits << " Hits" << std::endl;
#endif

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

  // in principle we can use "nhits" to heuristically dimension the workspace...
  this->device_isOuterHitOfCell_ =
      cms::cuda::make_device_unique<OuterHitOfCellContainer[]>(std::max(1, nhits - hh.offsetBPIX2()), stream);
  assert(this->device_isOuterHitOfCell_.get());

  this->isOuterHitOfCell_ = OuterHitOfCell{this->device_isOuterHitOfCell_.get(), hh.offsetBPIX2()};

  this->cellStorage_ =
      cms::cuda::make_device_unique<unsigned char[]>(TrackerTraits::maxNumOfActiveDoublets * sizeof(CellNeighbors) +
                                                         TrackerTraits::maxNumOfActiveDoublets * sizeof(CellTracks),
                                                     stream);
  this->device_theCellNeighborsContainer_ = (CellNeighbors *)this->cellStorage_.get();
  this->device_theCellTracksContainer_ =
      (CellTracks *)(this->cellStorage_.get() + TrackerTraits::maxNumOfActiveDoublets * sizeof(CellNeighbors));

  {
    int threadsPerBlock = 128;
    // at least one block!
    int blocks = (std::max(1, nhits - hh.offsetBPIX2()) + threadsPerBlock - 1) / threadsPerBlock;
    initDoublets<TrackerTraits><<<blocks, threadsPerBlock, 0, stream>>>(this->isOuterHitOfCell_,
                                                                        nhits,
                                                                        this->device_theCellNeighbors_.get(),
                                                                        this->device_theCellNeighborsContainer_,
                                                                        this->device_theCellTracks_.get(),
                                                                        this->device_theCellTracksContainer_);
    cudaCheck(cudaGetLastError());
  }

  this->device_theCells_ =
      cms::cuda::make_device_unique<GPUCACell[]>(this->params_.cellCuts_.maxNumberOfDoublets_, stream);

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

  if (0 == nhits)
    return;  // protect against empty events

  // take all layer pairs into account
  auto nActualPairs = this->params_.nPairs();

  int stride = 4;
  int threadsPerBlock = TrackerTraits::getDoubletsFromHistoMaxBlockSize / stride;
  int blocks = (4 * nhits + threadsPerBlock - 1) / threadsPerBlock;
  dim3 blks(1, blocks, 1);
  dim3 thrs(stride, threadsPerBlock, 1);
  getDoubletsFromHisto<TrackerTraits><<<blks, thrs, 0, stream>>>(this->device_theCells_.get(),
                                                                 this->device_nCells_,
                                                                 this->device_theCellNeighbors_.get(),
                                                                 this->device_theCellTracks_.get(),
                                                                 hh.view(),
                                                                 this->isOuterHitOfCell_,
                                                                 nActualPairs,
                                                                 this->params_.cellCuts_);
  cudaCheck(cudaGetLastError());

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif
}

template <typename TrackerTraits>
void CAHitNtupletGeneratorKernelsGPU<TrackerTraits>::classifyTuples(HitsOnCPU const &hh,
                                                                    TkSoA *tracks_d,
                                                                    cudaStream_t cudaStream) {
  using namespace caHitNtupletGeneratorKernels;

  // these are pointer on GPU!
  auto const *tuples_d = &tracks_d->hitIndices;
  auto *quality_d = tracks_d->qualityData();

  int32_t nhits = hh.nHits();

  auto blockSize = 64;

  // classify tracks based on kinematics
  auto numberOfBlocks = this->nQuadrupletBlocks(blockSize);
  kernel_classifyTracks<TrackerTraits>
      <<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples_d, tracks_d, this->params_.qualityCuts_, quality_d);

  if (this->params_.lateFishbone_) {
    // apply fishbone cleaning to good tracks
    numberOfBlocks = this->nDoubletBlocks(blockSize);
    kernel_fishboneCleaner<TrackerTraits>
        <<<numberOfBlocks, blockSize, 0, cudaStream>>>(this->device_theCells_.get(), this->device_nCells_, quality_d);
    cudaCheck(cudaGetLastError());
  }

  // mark duplicates (tracks that share a doublet)
  numberOfBlocks = this->nDoubletBlocks(blockSize);
  kernel_fastDuplicateRemover<TrackerTraits><<<numberOfBlocks, blockSize, 0, cudaStream>>>(
      this->device_theCells_.get(), this->device_nCells_, tracks_d, this->params_.dupPassThrough_);
  cudaCheck(cudaGetLastError());
#ifdef GPU_DEBUG
  cudaCheck(cudaDeviceSynchronize());
#endif

  if (this->params_.doSharedHitCut_ || this->params_.doStats_) {
    // fill hit->track "map"
    assert(this->hitToTupleView_.offSize > nhits);
    numberOfBlocks = this->nQuadrupletBlocks(blockSize);
    kernel_countHitInTracks<TrackerTraits>
        <<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples_d, quality_d, this->device_hitToTuple_.get());
    cudaCheck(cudaGetLastError());
    assert((this->hitToTupleView_.assoc == this->device_hitToTuple_.get()) &&
           (this->hitToTupleView_.offStorage == this->device_hitToTupleStorage_.get()) &&
           (this->hitToTupleView_.offSize > 0));
    cms::cuda::launchFinalize(this->hitToTupleView_, cudaStream);
    cudaCheck(cudaGetLastError());
    kernel_fillHitInTracks<TrackerTraits>
        <<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples_d, quality_d, this->device_hitToTuple_.get());
    cudaCheck(cudaGetLastError());
#ifdef GPU_DEBUG
    cudaCheck(cudaDeviceSynchronize());
#endif
  }

  if (this->params_.doSharedHitCut_) {
    // mark duplicates (tracks that share at least one hit)
    numberOfBlocks = (this->hitToTupleView_.offSize + blockSize - 1) / blockSize;

    kernel_rejectDuplicate<TrackerTraits>
        <<<numberOfBlocks, blockSize, 0, cudaStream>>>(tracks_d,
                                                       quality_d,
                                                       this->params_.minHitsForSharingCut_,
                                                       this->params_.dupPassThrough_,
                                                       this->device_hitToTuple_.get());

    kernel_sharedHitCleaner<TrackerTraits>
        <<<numberOfBlocks, blockSize, 0, cudaStream>>>(hh.view(),
                                                       tracks_d,
                                                       quality_d,
                                                       this->params_.minHitsForSharingCut_,
                                                       this->params_.dupPassThrough_,
                                                       this->device_hitToTuple_.get());

    if (this->params_.useSimpleTripletCleaner_) {
      kernel_simpleTripletCleaner<TrackerTraits>
          <<<numberOfBlocks, blockSize, 0, cudaStream>>>(tracks_d,
                                                         quality_d,
                                                         this->params_.minHitsForSharingCut_,
                                                         this->params_.dupPassThrough_,
                                                         this->device_hitToTuple_.get());
    } else {
      kernel_tripletCleaner<TrackerTraits>
          <<<numberOfBlocks, blockSize, 0, cudaStream>>>(tracks_d,
                                                         quality_d,
                                                         this->params_.minHitsForSharingCut_,
                                                         this->params_.dupPassThrough_,
                                                         this->device_hitToTuple_.get());
    }
    cudaCheck(cudaGetLastError());
#ifdef GPU_DEBUG
    cudaCheck(cudaDeviceSynchronize());
#endif
  }

  if (this->params_.doStats_) {
    numberOfBlocks = (std::max(nhits, int(this->params_.cellCuts_.maxNumberOfDoublets_)) + blockSize - 1) / blockSize;
    kernel_checkOverflows<TrackerTraits>
        <<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples_d,
                                                       this->device_tupleMultiplicity_.get(),
                                                       this->device_hitToTuple_.get(),
                                                       this->device_hitTuple_apc_,
                                                       this->device_theCells_.get(),
                                                       this->device_nCells_,
                                                       this->device_theCellNeighbors_.get(),
                                                       this->device_theCellTracks_.get(),
                                                       this->isOuterHitOfCell_,
                                                       nhits,
                                                       this->params_.cellCuts_.maxNumberOfDoublets_,
                                                       this->counters_);
    cudaCheck(cudaGetLastError());
  }

  if (this->params_.doStats_) {
    // counters (add flag???)
    numberOfBlocks = (this->hitToTupleView_.offSize + blockSize - 1) / blockSize;
    kernel_doStatsForHitInTracks<TrackerTraits>
        <<<numberOfBlocks, blockSize, 0, cudaStream>>>(this->device_hitToTuple_.get(), this->counters_);
    cudaCheck(cudaGetLastError());
    numberOfBlocks = (3 * TrackerTraits::maxNumberOfQuadruplets / 4 + blockSize - 1) / blockSize;
    kernel_doStatsForTracks<TrackerTraits>
        <<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples_d, quality_d, this->counters_);
    cudaCheck(cudaGetLastError());
  }
#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

#ifdef DUMP_GPU_TK_TUPLES
  static std::atomic<int> iev(0);
  static std::mutex lock;
  {
    std::lock_guard<std::mutex> guard(lock);
    ++iev;
    for (int k = 0; k < 20000; k += 500) {
      kernel_print_found_ntuplets<TrackerTraits><<<1, 32, 0, cudaStream>>>(
          hh.view(), tuples_d, tracks_d, quality_d, this->device_hitToTuple_.get(), k, k + 500, iev);
      cudaDeviceSynchronize();
    }
    kernel_print_found_ntuplets<TrackerTraits><<<1, 32, 0, cudaStream>>>(
        hh.view(), tuples_d, tracks_d, quality_d, this->device_hitToTuple_.get(), 20000, 1000000, iev);
    cudaDeviceSynchronize();
    // cudaStreamSynchronize(cudaStream);
  }
#endif
}

template <typename TrackerTraits>
void CAHitNtupletGeneratorKernelsGPU<TrackerTraits>::printCounters(Counters const *counters) {
  caHitNtupletGeneratorKernels::kernel_printCounters<<<1, 1>>>(counters);
}

template class CAHitNtupletGeneratorKernelsGPU<pixelTopology::Phase1>;
template class CAHitNtupletGeneratorKernelsGPU<pixelTopology::Phase2>;
