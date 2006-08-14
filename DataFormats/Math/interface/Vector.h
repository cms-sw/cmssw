#ifndef TrackReco_Vector_h
#define TrackReco_Vector_h
// $Id: Vector.h,v 1.3 2006/03/31 10:39:42 llista Exp $
#include <Rtypes.h>
#include <Math/SVector.h>

namespace math {
  /// fixed size vector
  template<unsigned int N>
  struct Vector {
    typedef ROOT::Math::SVector<Double32_t, N> type;
  };

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
}

#endif
