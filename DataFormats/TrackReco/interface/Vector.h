#ifndef TrackReco_Vector_h
#define TrackReco_Vector_h
// $Id: Vector.h,v 1.1 2005/11/22 13:51:44 llista Exp $
//
// very simple persistent vector with minumal functionalities
//
#include <algorithm>
#include <Rtypes.h>

namespace reco {

  template< unsigned int N > 
  struct Vector {
    enum { dimension = N };
    typedef unsigned int index;
    Vector() { }
    Vector( const Vector & o ) {
      std::copy( o.val, o.val + dimension, val );
    }
    Vector( const double * v ) {
      std::copy( v, v + dimension, val );
    }
    double & operator()( index i ) { return val[ i ]; }
    const double & operator()( index i ) const { return val[ i ]; }
    template< index i >
    double & get() { return val[ i ]; }
    template< index i >
    double get() const { return val[ i ]; }
    
  private:
    Double32_t val[ dimension ];
  };

  typedef Vector<3> Vector3D;  
}

template<unsigned int n>
bool operator==( const reco::Vector<n> & a, const reco::Vector<n> & b ) {
  for( unsigned int i = 0; i < n; ++i )
    if ( a( i ) != b( i ) ) return false;
  return true;
}

#endif
