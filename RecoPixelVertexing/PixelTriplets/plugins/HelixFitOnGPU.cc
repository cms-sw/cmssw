#include "HelixFitOnGPU.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

void HelixFitOnGPU::allocateOnGPU(TuplesOnGPU::Container const * tuples, TupleMultiplicity const * tupleMultiplicity, Rfit::helix_fit * helix_fit_results) {

  tuples_d = tuples;
  tupleMultiplicity_d = tupleMultiplicity;
  helix_fit_results_d = helix_fit_results;

  assert(tuples_d); assert(tupleMultiplicity_d); assert(helix_fit_results_d);

  cudaCheck(cudaMalloc(&hitsGPU_, maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix3xNd<4>)));
  cudaCheck(cudaMemset(hitsGPU_, 0x00, maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix3xNd<4>)));

  cudaCheck(cudaMalloc(&hits_geGPU_, maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix6x4f)));
  cudaCheck(cudaMemset(hits_geGPU_, 0x00, maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix6x4f)));

  cudaCheck(cudaMalloc(&fast_fit_resultsGPU_, maxNumberOfConcurrentFits_ * sizeof(Rfit::Vector4d)));
  cudaCheck(cudaMemset(fast_fit_resultsGPU_, 0x00, maxNumberOfConcurrentFits_ * sizeof(Rfit::Vector4d)));

  cudaCheck(cudaMalloc(&circle_fit_resultsGPU_, maxNumberOfConcurrentFits_ * sizeof(Rfit::circle_fit)));
  cudaCheck(cudaMemset(circle_fit_resultsGPU_, 0x00, maxNumberOfConcurrentFits_ * sizeof(Rfit::circle_fit)));

}

void HelixFitOnGPU::deallocateOnGPU() {

  cudaFree(hitsGPU_);
  cudaFree(hits_geGPU_);
  cudaFree(fast_fit_resultsGPU_);
  cudaFree(circle_fit_resultsGPU_);

}



