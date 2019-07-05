#include "CAHitNtupletGeneratorKernels.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

void CAHitNtupletGeneratorKernels::allocateOnGPU(cuda::stream_t<>& stream) {
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

  device_hitToTuple_ = cs->make_device_unique<HitToTuple>(stream);

  device_tupleMultiplicity_ = cs->make_device_unique<TupleMultiplicity>(stream);

  auto storageSize = 3+(std::max(TupleMultiplicity::wsSize(), HitToTuple::wsSize())+sizeof(AtomicPairCounter::c_type))/sizeof(AtomicPairCounter::c_type);

  device_storage_ = cs->make_device_unique<AtomicPairCounter::c_type[]>(storageSize,stream);
  
  device_hitTuple_apc_ = (AtomicPairCounter*)device_storage_.get();
  device_hitToTuple_apc_ = (AtomicPairCounter*)device_storage_.get()+1;
  device_nCells_ = (uint32_t *)(device_storage_.get()+2);
  device_tmws_ = (uint8_t*)(device_storage_.get()+3);

  assert(device_tmws_+std::max(TupleMultiplicity::wsSize(), HitToTuple::wsSize()) <= (uint8_t*)(device_storage_.get()+storageSize));

  cudaCheck(cudaMemsetAsync(device_nCells_, 0, sizeof(uint32_t), stream.id()));
  cudautils::launchZero(device_tupleMultiplicity_.get(), stream.id());
  cudautils::launchZero(device_hitToTuple_.get(), stream.id());  // we may wish to keep it in the edm...
}

