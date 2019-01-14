#include "CAHitQuadrupletGeneratorKernels.h"


void
CAHitQuadrupletGeneratorKernels::deallocateOnGPU()
{

  cudaFree(device_theCells_);
  cudaFree(device_isOuterHitOfCell_);
  cudaFree(device_nCells_);
//  cudaFree(device_hitToTuple_);
  cudaFree(device_hitToTuple_apc_);

}

void CAHitQuadrupletGeneratorKernels::allocateOnGPU()
{
  //////////////////////////////////////////////////////////
  // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
  //////////////////////////////////////////////////////////

  cudaCheck(cudaMalloc(&device_theCells_,
             CAConstants::maxNumberOfLayerPairs() * CAConstants::maxNumberOfDoublets() * sizeof(GPUCACell)));
  cudaCheck(cudaMalloc(&device_nCells_, sizeof(uint32_t)));
  cudaCheck(cudaMemset(device_nCells_, 0, sizeof(uint32_t)));

  cudaCheck(cudaMalloc(&device_isOuterHitOfCell_,
             PixelGPUConstants::maxNumberOfHits * sizeof(CAConstants::OuterHitOfCell)));
  cudaCheck(cudaMemset(device_isOuterHitOfCell_, 0,
             PixelGPUConstants::maxNumberOfHits * sizeof(CAConstants::OuterHitOfCell)));

//   cudaCheck(cudaMalloc(&device_hitToTuple_, sizeof(HitToTuple)));
   cudaCheck(cudaMalloc(&device_hitToTuple_apc_, sizeof(AtomicPairCounter)));

}

void CAHitQuadrupletGeneratorKernels::cleanup(cudaStream_t cudaStream) {
  // this lazily resets temporary memory for the next event, and is not needed for reading the output
  cudaCheck(cudaMemsetAsync(device_isOuterHitOfCell_, 0,
                            PixelGPUConstants::maxNumberOfHits * sizeof(CAConstants::OuterHitOfCell),
                            cudaStream));
  cudaCheck(cudaMemsetAsync(device_nCells_, 0, sizeof(uint32_t), cudaStream));
}

