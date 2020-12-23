#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "CAHitNtupletGeneratorKernels.h"

template <>
#ifdef __CUDACC__
void CAHitNtupletGeneratorKernelsGPU::allocateOnGPU(cudaStream_t stream) {
#else
void CAHitNtupletGeneratorKernelsCPU::allocateOnGPU(cudaStream_t stream) {
#endif
  //////////////////////////////////////////////////////////
  // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
  //////////////////////////////////////////////////////////

  device_theCellNeighbors_ = Traits::template make_unique<CAConstants::CellNeighborsVector>(stream);
  device_theCellTracks_ = Traits::template make_unique<CAConstants::CellTracksVector>(stream);

  device_hitToTuple_ = Traits::template make_unique<HitToTuple>(stream);

  device_tupleMultiplicity_ = Traits::template make_unique<TupleMultiplicity>(stream);

  device_storage_ = Traits::template make_unique<cms::cuda::AtomicPairCounter::c_type[]>(3, stream);

  device_hitTuple_apc_ = (cms::cuda::AtomicPairCounter*)device_storage_.get();
  device_hitToTuple_apc_ = (cms::cuda::AtomicPairCounter*)device_storage_.get() + 1;
  device_nCells_ = (uint32_t*)(device_storage_.get() + 2);

  if constexpr (std::is_same<Traits, cms::cudacompat::GPUTraits>::value) {
    cudaCheck(cudaMemsetAsync(device_nCells_, 0, sizeof(uint32_t), stream));
  } else {
    *device_nCells_ = 0;
  }
  cms::cuda::launchZero(device_tupleMultiplicity_.get(), stream);
  cms::cuda::launchZero(device_hitToTuple_.get(), stream);  // we may wish to keep it in the edm...
}
