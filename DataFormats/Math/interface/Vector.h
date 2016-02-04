#ifndef TrackReco_Vector_h
#define TrackReco_Vector_h
// $Id: Vector.h,v 1.6 2006/11/20 09:06:52 llista Exp $
#include "Rtypes.h"
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
