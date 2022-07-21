#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "CAHitNtupletGeneratorKernels.h"

//#define GPU_DEBUG
template <typename TrackerTraits>
#ifdef __CUDACC__
void CAHitNtupletGeneratorKernelsGPU<TrackerTraits>::allocateOnGPU(int32_t nHits, cudaStream_t stream) {
  using Traits = cms::cudacompat::GPUTraits;
#else
void CAHitNtupletGeneratorKernelsCPU<TrackerTraits>::allocateOnGPU(int32_t nHits, cudaStream_t stream) {
  using Traits = cms::cudacompat::CPUTraits;
#endif
  //////////////////////////////////////////////////////////
  // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
  //////////////////////////////////////////////////////////

  this->device_theCellNeighbors_ = Traits::template make_unique<CellNeighborsVector>(stream);
  this->device_theCellTracks_ = Traits::template make_unique<CellTracksVector>(stream);

#ifdef GPU_DEBUG
  std::cout << "Allocation for tuple building. N hits " << nHits << std::endl;
#endif

  nHits++;  // storage requires one more counter;
  assert(nHits > 0);
  this->device_hitToTuple_ = Traits::template make_unique<HitToTuple>(stream);
  this->device_hitToTupleStorage_ = Traits::template make_unique<typename HitToTuple::Counter[]>(nHits, stream);
  this->hitToTupleView_.assoc = this->device_hitToTuple_.get();
  this->hitToTupleView_.offStorage = this->device_hitToTupleStorage_.get();
  this->hitToTupleView_.offSize = nHits;

  this->device_tupleMultiplicity_ = Traits::template make_unique<TupleMultiplicity>(stream);

  this->device_storage_ = Traits::template make_unique<cms::cuda::AtomicPairCounter::c_type[]>(3, stream);

  this->device_hitTuple_apc_ = (cms::cuda::AtomicPairCounter*)this->device_storage_.get();
  this->device_hitToTuple_apc_ = (cms::cuda::AtomicPairCounter*)this->device_storage_.get() + 1;
  this->device_nCells_ = (uint32_t*)(this->device_storage_.get() + 2);

  // FIXME: consider collapsing these 3 in one adhoc kernel
  if constexpr (std::is_same<Traits, cms::cudacompat::GPUTraits>::value) {
    cudaCheck(cudaMemsetAsync(this->device_nCells_, 0, sizeof(uint32_t), stream));
  } else {
    *(this->device_nCells_) = 0;
  }
  cms::cuda::launchZero(this->device_tupleMultiplicity_.get(), stream);
  cms::cuda::launchZero(this->hitToTupleView_, stream);  // we may wish to keep it in the edm
#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif
}

template class CAHitNtupletGeneratorKernelsGPU<pixelTopology::Phase1>;
template class CAHitNtupletGeneratorKernelsGPU<pixelTopology::Phase2>;

template class CAHitNtupletGeneratorKernelsCPU<pixelTopology::Phase1>;
template class CAHitNtupletGeneratorKernelsCPU<pixelTopology::Phase2>;
