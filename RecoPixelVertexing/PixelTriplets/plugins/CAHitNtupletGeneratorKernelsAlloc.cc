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

  memoryPool::Deleter deleter = memoryPool::Deleter(std::make_shared<memoryPool::cuda::BundleDelete>(stream, where));

  device_theCellNeighbors_ = memoryPool::cuda::makeBuffer<caConstants::CellNeighborsVector>(1, deleter);
  device_theCellTracks_ = memoryPool::cuda::makeBuffer<caConstants::CellTracksVector>(1, deleter);

#ifdef GPU_DEBUG
  std::cout << "Allocation for tuple building. N hits " << nHits
            << ((where == memoryPool::onDevice) ? " on GPU" : " on CPU") << std::endl;
#endif

  nHits++;  // storage requires one more counter;
  assert(nHits > 0);
  device_hitToTuple_ = memoryPool::cuda::makeBuffer<HitToTuple>(1, deleter);
  device_hitToTupleStorage_ = memoryPool::cuda::makeBuffer<HitToTuple::Counter>(nHits, deleter);
  hitToTupleView_.assoc = device_hitToTuple_.get();
  hitToTupleView_.offStorage = device_hitToTupleStorage_.get();
  hitToTupleView_.offSize = nHits;

  device_tupleMultiplicity_ = memoryPool::cuda::makeBuffer<TupleMultiplicity>(1, deleter);

  device_storage_ = memoryPool::cuda::makeBuffer<cms::cuda::AtomicPairCounter::c_type>(3, deleter);

  device_hitTuple_apc_ = (cms::cuda::AtomicPairCounter*)device_storage_.get();
  device_hitToTuple_apc_ = (cms::cuda::AtomicPairCounter*)device_storage_.get() + 1;
  device_nCells_ = (uint32_t*)(device_storage_.get() + 2);

  // FIXME: consider collapsing these 3 in one adhoc kernel
  if constexpr (where == memoryPool::onDevice) {
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
