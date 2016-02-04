#ifndef TrackReco_Error_h
#define TrackReco_Error_h
// $Id: Error.h,v 1.10 2006/11/20 09:06:52 llista Exp $
//
// Symmetric matrix
//
#include "Rtypes.h"
#include "Math/SMatrix.h"
#include "Math/BinaryOperators.h"

#include <vector>

namespace math {

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

  /// fixed size error matrix
  template<unsigned int N>
  struct Error {
    typedef typename ErrorD<N>::type type;
  };

}

#endif
