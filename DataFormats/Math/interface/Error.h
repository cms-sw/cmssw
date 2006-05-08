#ifndef TrackReco_Error_h
#define TrackReco_Error_h
// $Id: Error.h,v 1.4 2006/05/08 08:07:00 llista Exp $
//
// very simple persistent error matrix with minumal functionalities
//
#include <Rtypes.h>
#include <Math/SMatrix.h>
#include <Math/BinaryOperators.h>

namespace math {
  /// fixed size error matrix
  template<unsigned int N>
  struct Error {
    typedef ROOT::Math::SMatrix<Double32_t, N, N, ROOT::Math::MatRepSym<Double32_t, N> > type;
  };

  /// fixed size error matrix with double components
  template<unsigned int N>
  struct ErrorD {
    typedef ROOT::Math::SMatrix<double, N, N, ROOT::Math::MatRepSym<double, N> > type;
  };

  /// fixed size error matrix with double components
  template<unsigned int N>
  struct ErrorF {
    typedef ROOT::Math::SMatrix<float, N, N, ROOT::Math::MatRepSym<float, N> > type;
  };
}

#endif
