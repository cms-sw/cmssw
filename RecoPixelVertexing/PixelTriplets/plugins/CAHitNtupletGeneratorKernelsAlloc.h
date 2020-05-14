#include "CAHitNtupletGeneratorKernels.h"

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

template <>
#ifdef __CUDACC__
void CAHitNtupletGeneratorKernelsGPU::allocateOnGPU(cudaStream_t stream) {
#else
void CAHitNtupletGeneratorKernelsCPU::allocateOnGPU(cudaStream_t stream) {
#endif
  //////////////////////////////////////////////////////////
  // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
  //////////////////////////////////////////////////////////

  /* not used at the moment 
  cudaCheck(cudaMalloc(&device_theCellNeighbors_, sizeof(CAConstants::CellNeighborsVector)));
  cudaCheck(cudaMemset(device_theCellNeighbors_, 0, sizeof(CAConstants::CellNeighborsVector)));
  cudaCheck(cudaMalloc(&device_theCellTracks_, sizeof(CAConstants::CellTracksVector)));
  cudaCheck(cudaMemset(device_theCellTracks_, 0, sizeof(CAConstants::CellTracksVector)));
  */

  device_hitToTuple_ = Traits::template make_unique<HitToTuple>(stream);

  device_tupleMultiplicity_ = Traits::template make_unique<TupleMultiplicity>(stream);

  auto storageSize =
      3 + (std::max(TupleMultiplicity::wsSize(), HitToTuple::wsSize()) + sizeof(cms::cuda::AtomicPairCounter::c_type)) /
              sizeof(cms::cuda::AtomicPairCounter::c_type);

  device_storage_ = Traits::template make_unique<cms::cuda::AtomicPairCounter::c_type[]>(storageSize, stream);

  device_hitTuple_apc_ = (cms::cuda::AtomicPairCounter*)device_storage_.get();
  device_hitToTuple_apc_ = (cms::cuda::AtomicPairCounter*)device_storage_.get() + 1;
  device_nCells_ = (uint32_t*)(device_storage_.get() + 2);
  device_tmws_ = (uint8_t*)(device_storage_.get() + 3);

  assert(device_tmws_ + std::max(TupleMultiplicity::wsSize(), HitToTuple::wsSize()) <=
         (uint8_t*)(device_storage_.get() + storageSize));

  if
#ifndef __CUDACC__
      constexpr
#endif
      (std::is_same<Traits, cms::cudacompat::GPUTraits>::value) {
    cudaCheck(cudaMemsetAsync(device_nCells_, 0, sizeof(uint32_t), stream));
  } else {
    *device_nCells_ = 0;
  }
  cms::cuda::launchZero(device_tupleMultiplicity_.get(), stream);
  cms::cuda::launchZero(device_hitToTuple_.get(), stream);  // we may wish to keep it in the edm...
}
