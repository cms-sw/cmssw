#ifndef TrackReco_Vector_h
#define TrackReco_Vector_h
// $Id: Vector.h,v 1.2 2006/03/31 09:24:49 llista Exp $
#include <Rtypes.h>
#include <Math/SVector.h>

namespace math {
  /// fixed size vector
  template<unsigned int D>
  struct Vector {
    typedef ROOT::Math::SVector<Double32_t, D> type;
  };
}

#endif
