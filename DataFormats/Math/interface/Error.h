#ifndef TrackReco_Error_h
#define TrackReco_Error_h
// $Id: Error.h,v 1.7 2006/05/24 12:49:07 llista Exp $
//
// very simple persistent error matrix with minumal functionalities
//
#include <Rtypes.h>
#include <Math/SMatrix.h>
#include <Math/BinaryOperators.h>

#include <vector>

namespace math {

  template< typename D, unsigned int N >
  struct ErrorMatrix {
    enum { dimension = N, size = N * ( N + 1 ) / 2 };
    enum { kRows = N, kCols = N, kSize = size };
    typedef unsigned int index;
    ErrorMatrix() { err.resize(size);}
    ErrorMatrix( const ErrorMatrix<D, N> & o ) {
      //      err.resize(size);
      std::copy( o.err.begin(), o.err.end(), back_inserter(err) );
    }
    ErrorMatrix( const D * v ) {
      //      err.resize(size);
      for (unsigned int i=0; i<size; i++) err.push_back(v[i]);
    }
    ErrorMatrix<D, N> & operator=( const ErrorMatrix<D, N> & o ) {
      std::copy( o.err.begin(), o.err.end(), back_inserter(err)  );
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
//    D err[ size ];
      std::vector<D> err;
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
