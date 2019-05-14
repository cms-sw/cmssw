#include "CAHitQuadrupletGeneratorKernels.h"

void CAHitQuadrupletGeneratorKernels::deallocateOnGPU() {
  if (doStats_) {
    // crash on multi-gpu processes
    printCounters();
  }
  cudaFree(counters_);

  cudaFree(device_nCells_);
  cudaFree(device_theCellNeighbors_);
  cudaFree(device_theCellTracks_);

  cudaFree(device_hitToTuple_);
  cudaFree(device_hitToTuple_apc_);
  cudaFree(device_tupleMultiplicity_);
  cudaFree(device_tmws_);
}

void CAHitQuadrupletGeneratorKernels::allocateOnGPU() {
  //////////////////////////////////////////////////////////
  // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
  //////////////////////////////////////////////////////////

  cudaCheck(cudaMalloc(&counters_, sizeof(Counters)));
  cudaCheck(cudaMemset(counters_, 0, sizeof(Counters)));

  cudaCheck(cudaMalloc(&device_nCells_, sizeof(uint32_t)));
  cudaCheck(cudaMemset(device_nCells_, 0, sizeof(uint32_t)));

  cudaCheck(cudaMalloc(&device_theCellNeighbors_, sizeof(CAConstants::CellNeighborsVector)));
  cudaCheck(cudaMemset(device_theCellNeighbors_, 0, sizeof(CAConstants::CellNeighborsVector)));
  cudaCheck(cudaMalloc(&device_theCellTracks_, sizeof(CAConstants::CellTracksVector)));
  cudaCheck(cudaMemset(device_theCellTracks_, 0, sizeof(CAConstants::CellTracksVector)));

  cudaCheck(cudaMalloc(&device_hitToTuple_, sizeof(HitToTuple)));
  cudaCheck(cudaMemset(device_hitToTuple_, 0, sizeof(HitToTuple)));  // overkill
  cudaCheck(cudaMalloc(&device_hitToTuple_apc_, sizeof(AtomicPairCounter)));

  cudaCheck(cudaMalloc(&device_tupleMultiplicity_, sizeof(TupleMultiplicity)));
  cudaCheck(cudaMemset(device_tupleMultiplicity_, 0, sizeof(TupleMultiplicity)));  // overkill

  cudaCheck(cudaMalloc(&device_tmws_, std::max(TupleMultiplicity::wsSize(), HitToTuple::wsSize())));
}

void CAHitQuadrupletGeneratorKernels::cleanup(cudaStream_t cudaStream) {
#ifdef GPU_DEBUG
  std::cout << "CAHitQuadrupletGeneratorKernels::cleanup" << std::endl;
#endif

  // this lazily resets temporary memory for the next event, and is not needed for reading the output
  device_theCells_ = nullptr;
  device_isOuterHitOfCell_ = nullptr;
  device_theCellNeighborsContainer_ = nullptr;
  device_theCellTracksContainer_ = nullptr;

  cudaCheck(cudaMemsetAsync(device_nCells_, 0, sizeof(uint32_t), cudaStream));

  cudautils::launchZero(device_tupleMultiplicity_, cudaStream);

  cudautils::launchZero(device_hitToTuple_, cudaStream);  // we may wish to keep it in the edm...
}
