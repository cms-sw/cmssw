#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HelixFitOnGPU.h"

void HelixFitOnGPU::allocateOnGPU(Tuples const *tuples,
                                  TupleMultiplicity const *tupleMultiplicity,
                                  OutputSoA *helix_fit_results) {
  tuples_ = tuples;
  tupleMultiplicity_ = tupleMultiplicity;
  outputSoa_ = helix_fit_results;

  assert(tuples_);
  assert(tupleMultiplicity_);
  assert(outputSoa_);
}

void HelixFitOnGPU::deallocateOnGPU() {}
