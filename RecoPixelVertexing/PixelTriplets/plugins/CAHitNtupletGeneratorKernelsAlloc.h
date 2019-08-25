#include "CAHitNtupletGeneratorKernels.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"


template<>
#ifdef __CUDACC__
void CAHitNtupletGeneratorKernelsGPU::allocateOnGPU(cuda::stream_t<>& stream) {
#else
void CAHitNtupletGeneratorKernelsCPU::allocateOnGPU(cuda::stream_t<>& stream) {
#endif
  //////////////////////////////////////////////////////////
  // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
  //////////////////////////////////////////////////////////

  edm::Service<CUDAService> cs;

  /* not used at the moment 
  cudaCheck(cudaMalloc(&device_theCellNeighbors_, sizeof(CAConstants::CellNeighborsVector)));
  cudaCheck(cudaMemset(device_theCellNeighbors_, 0, sizeof(CAConstants::CellNeighborsVector)));
  cudaCheck(cudaMalloc(&device_theCellTracks_, sizeof(CAConstants::CellTracksVector)));
  cudaCheck(cudaMemset(device_theCellTracks_, 0, sizeof(CAConstants::CellTracksVector)));
  */

  device_hitToTuple_ = Traits:: template make_unique<HitToTuple>(cs,stream);

  device_tupleMultiplicity_ = Traits:: template make_unique<TupleMultiplicity>(cs,stream);

  auto storageSize = 3+(std::max(TupleMultiplicity::wsSize(), HitToTuple::wsSize())+sizeof(AtomicPairCounter::c_type))/sizeof(AtomicPairCounter::c_type);

  device_storage_ = Traits:: template make_unique<AtomicPairCounter::c_type[]>(cs, storageSize,stream);
  
  device_hitTuple_apc_ = (AtomicPairCounter*)device_storage_.get();
  device_hitToTuple_apc_ = (AtomicPairCounter*)device_storage_.get()+1;
  device_nCells_ = (uint32_t *)(device_storage_.get()+2);
  device_tmws_ = (uint8_t*)(device_storage_.get()+3);

  assert(device_tmws_+std::max(TupleMultiplicity::wsSize(), HitToTuple::wsSize()) <= (uint8_t*)(device_storage_.get()+storageSize));

  if
#ifndef __CUDACC__
    constexpr
#endif
      (std::is_same<Traits,cudaCompat::GPUTraits>::value) {
    cudaCheck(cudaMemsetAsync(device_nCells_, 0, sizeof(uint32_t), stream.id()));
  }else {
     *device_nCells_ = 0;
  }  
  cudautils::launchZero(device_tupleMultiplicity_.get(), stream.id());
  cudautils::launchZero(device_hitToTuple_.get(), stream.id());  // we may wish to keep it in the edm...
}

