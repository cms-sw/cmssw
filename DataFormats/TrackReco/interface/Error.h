#ifndef TrackReco_Error_h
#define TrackReco_Error_h
// $Id: Error.h,v 1.1 2005/11/22 13:51:44 llista Exp $
//
// very simple persistent error matrix with minumal functionalities
//
#include <algorithm>
#include <Rtypes.h>

namespace reco {

  template< unsigned int N >
  struct Error {
    enum { dimension = N, size = N * ( N + 1 ) / 2 };
    typedef unsigned int index;
    Error() { }
    Error( const Error<N> & o ) { 
      std::copy( o.err, o.err + size, err ); 
    }
    Error( const double * v ) {
      std::copy( v, v + size, err );
    }
    Error<N> & operator=( const Error<N> & o ) {
      std::copy( o.err, o.err + size, err ); 
      return * this;
    }
    double & operator()( index i, index j ) { 
      return err[ idx( i, j ) ]; 
    }
    const double & operator()( index i, index j ) const { 
      return err[ idx( i, j ) ]; 
    }
    template< index i, index j >
    double & get() { 
      return err[ idx_t< i, j >::value ]; 
    }
    template< index i, index j >
    const double & get() const { 
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
    Double32_t err[ size ];
  };

  typedef Error<3> Error3D;

}

#endif
