#ifndef DataFormats_Math_Matrix
#define DataFormats_Math_Matrix
#include "Math/SMatrix.h"
#include "Math/SVector.h"

namespace math {
  template<unsigned int N, unsigned int M>
  struct Matrix
  {
    typedef ROOT::Math::SMatrix<double,N,M> type;
  };
}
#endif
