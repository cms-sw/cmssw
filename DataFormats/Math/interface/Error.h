#ifndef TrackReco_Error_h
#define TrackReco_Error_h
// $Id: Error.h,v 1.1 2005/12/15 03:39:02 llista Exp $
//
// very simple persistent error matrix with minumal functionalities
//
#include <Rtypes.h>
#include <Math/MatrixRepresentationsStatic.h>

namespace math {
  /// fixed size error matrix
  template<unsigned int D>
  struct Error {
    typedef ROOT::Math::MatRepSym<Double32_t, D> type;
  };
  /// fixed size error matrix with double components
  template<unsigned int D>
  struct ErrorD {
    typedef ROOT::Math::MatRepSym<double, D> type;
  };
}

#endif
