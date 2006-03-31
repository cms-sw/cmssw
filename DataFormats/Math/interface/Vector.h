#ifndef TrackReco_Vector_h
#define TrackReco_Vector_h
// $Id: Vector.h,v 1.1 2005/12/15 03:39:02 llista Exp $
#include <Rtypes.h>
#include <Math/SVector.h>

namespace math {
  template<unsigned int D>
  struct Vector {
    typedef ROOT::Math::SVector<Double32_t, D> type;
  };
}

#endif
