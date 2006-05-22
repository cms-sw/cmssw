#ifndef TrackReco_Error_h
#define TrackReco_Error_h
// $Id: Error.h,v 1.5 2006/05/08 08:58:45 llista Exp $
//
// very simple persistent error matrix with minumal functionalities
//
#include <Rtypes.h>
#include <Math/SMatrix.h>
#include <Math/BinaryOperators.h>

namespace math {

  template< typename D, unsigned int N >
  struct ErrorMatrix {
    enum { dimension = N, size = N * ( N + 1 ) / 2 };
    enum { kRows = N, kCols = N };
    typedef unsigned int index;
    ErrorMatrix() { }
    ErrorMatrix( const ErrorMatrix<D, N> & o ) {
      std::copy( o.err, o.err + size, err );
    }
    ErrorMatrix( const D * v ) {
      std::copy( v, v + size, err );
    }
    ErrorMatrix<D, N> & operator=( const ErrorMatrix<D, N> & o ) {
      std::copy( o.err, o.err + size, err );
      return * this;
    }
    D & operator()( index i, index j ) {
      return err[ idx( i, j ) ];
    }
    const D & operator()( index i, index j ) const {
      return err[ idx( i, j ) ];
    }
    template< index i, index j >
    D & get() {
      return err[ idx_t< i, j >::value ];
    }
    template< index i, index j >
    const D & get() const {
      return err[ idx_t< i, j >::value ];
    }
  private:
    index idx( index i, index j ) const {
      int a = ( i <= j ? i : j ), b = ( i <= j ? j : i );
      return a * dimension + b - a * ( a + 1 ) / 2;
    };
    template< index i, index j >
    struct idx_t {
      enum { a = ( i <= j ? i : j ), b = ( i <= j ? j : i ) };
      enum { value =  a * dimension + b - a * ( a + 1 ) / 2 };
    };
    D err[ size ];
  };

  /*
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
  */

  /// fixed size error matrix
  template<unsigned int N>
  struct Error {
    typedef ErrorMatrix<Double32_t, N> type;
  };

  /// fixed size error matrix with double components
  template<unsigned int N>
  struct ErrorD {
    typedef ErrorMatrix<double, N> type;
  };

  /// fixed size error matrix with double components
  template<unsigned int N>
  struct ErrorF {
    typedef ErrorMatrix<float, N> type;
  };
}

#endif
