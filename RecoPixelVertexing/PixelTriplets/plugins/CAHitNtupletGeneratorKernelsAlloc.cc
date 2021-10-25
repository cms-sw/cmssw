#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "CAHitNtupletGeneratorKernels.h"

template <>
#ifdef __CUDACC__
void CAHitNtupletGeneratorKernelsGPU::allocateOnGPU(int32_t nHits, cudaStream_t stream) {
#else
void CAHitNtupletGeneratorKernelsCPU::allocateOnGPU(int32_t nHits, cudaStream_t stream) {
#endif
  //////////////////////////////////////////////////////////
  // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
  //////////////////////////////////////////////////////////

  device_theCellNeighbors_ = Traits::template make_unique<caConstants::CellNeighborsVector>(stream);
  device_theCellTracks_ = Traits::template make_unique<caConstants::CellTracksVector>(stream);

#ifdef GPU_DEBUG
  std::cout << "Allocation for tuple building. N hits " << nHits << std::endl;
#endif

  nHits++;  // storage requires one more counter;
  assert(nHits > 0);
  device_hitToTuple_ = Traits::template make_unique<HitToTuple>(stream);
  device_hitToTupleStorage_ = Traits::template make_unique<HitToTuple::Counter[]>(nHits, stream);
  hitToTupleView_.assoc = device_hitToTuple_.get();
  hitToTupleView_.offStorage = device_hitToTupleStorage_.get();
  hitToTupleView_.offSize = nHits;

  device_tupleMultiplicity_ = Traits::template make_unique<TupleMultiplicity>(stream);

  device_storage_ = Traits::template make_unique<cms::cuda::AtomicPairCounter::c_type[]>(3, stream);

  device_hitTuple_apc_ = (cms::cuda::AtomicPairCounter*)device_storage_.get();
  device_hitToTuple_apc_ = (cms::cuda::AtomicPairCounter*)device_storage_.get() + 1;
  device_nCells_ = (uint32_t*)(device_storage_.get() + 2);

  // FIXME: consider collapsing these 3 in one adhoc kernel
  if constexpr (std::is_same<Traits, cms::cudacompat::GPUTraits>::value) {
    cudaCheck(cudaMemsetAsync(device_nCells_, 0, sizeof(uint32_t), stream));
  } else {
    *device_nCells_ = 0;
  }
  cms::cuda::launchZero(device_tupleMultiplicity_.get(), stream);
  cms::cuda::launchZero(hitToTupleView_, stream);  // we may wish to keep it in the edm
#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif
}
