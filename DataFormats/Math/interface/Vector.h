#ifndef TrackReco_Vector_h
#define TrackReco_Vector_h
#include "Rtypes.h"

#define SMATRIX_USE_CONSTEXPR
#include "Math/SVector.h"

namespace math {

  /// fixed size vector
  template <unsigned int N>
  struct VectorD {
    typedef ROOT::Math::SVector<double, N> type;
  };

  /// fixed size vector
  template <unsigned int N>
  struct VectorF {
    typedef ROOT::Math::SVector<float, N> type;
  };

  /// fixed size vector
  template <unsigned int N>
  struct Vector {
    typedef typename VectorD<N>::type type;
  };
}  // namespace math

#endif
