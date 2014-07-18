#ifndef TrackReco_Vector_h
#define TrackReco_Vector_h
#include "Rtypes.h"

#include "FWCore/Utilities/interface/GCC11Compatibility.h"

#ifdef CMS_NOCXX11
#define SMATRIX_USE_COMPUTATION
#else
#define SMATRIX_USE_CONSTEXPR
#endif

#include "Math/SVector.h"

namespace math {

  /// fixed size vector
  template<unsigned int N>
  struct VectorD {
    typedef ROOT::Math::SVector<double, N> type;
  };

  /// fixed size vector
  template<unsigned int N>
  struct VectorF {
    typedef ROOT::Math::SVector<float, N> type;
  };

  /// fixed size vector
  template<unsigned int N>
  struct Vector {
    typedef typename VectorD<N>::type type;
  };
}

#endif
