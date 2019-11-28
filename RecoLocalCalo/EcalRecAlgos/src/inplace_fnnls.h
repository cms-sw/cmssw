#ifndef inplace_fnnls_hpp
#define inpalce_fnnls_hpp

#include "RecoLocalCalo/EcalRecAlgos/interface/EigenMatrixTypes_gpu.h"

namespace ecal {
  namespace multifit {

    using matrix_t = SampleMatrix;
    using vector_t = SampleVector;

    __device__ bool inplace_fnnls(matrix_t const& A,
                                  vector_t const& b,
                                  vector_t& x,
                                  int& npassive,
                                  BXVectorType& activeBXs,
                                  PulseMatrixType& pulse_matrix,
                                  const double eps = 1e-11,
                                  const unsigned int max_iterations = 500);

  }  // namespace multifit
}  // namespace ecal

#endif
