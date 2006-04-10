#ifndef TrackReco_Error_h
#define TrackReco_Error_h
// $Id: Error.h,v 1.2 2006/03/31 10:39:42 llista Exp $
//
// very simple persistent error matrix with minumal functionalities
//
#include <Rtypes.h>
#include <Math/MatrixRepresentationsStatic.h>

namespace math {
  /// fixed size error matrix
  template<unsigned int N>
  struct Error {
    typedef ROOT::Math::MatRepSym<Double32_t, N> type;
  };

  /// fixed size error matrix with double components
  template<unsigned int N>
  struct ErrorD {
    typedef ROOT::Math::MatRepSym<double, N> type;
  };

  /// fixed size error matrix with double components
  template<unsigned int N>
  struct ErrorF {
    typedef ROOT::Math::MatRepSym<float, N> type;
  };
}

#endif
