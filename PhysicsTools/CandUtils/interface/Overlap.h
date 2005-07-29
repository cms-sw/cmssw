#ifndef PHYSICSTOOLS_OVERLAP_H
#define PHYSICSTOOLS_OVERLAP_H
// $Id: Overlap.h,v 1.4 2005/07/07 13:52:27 llista Exp $
#include "PhysicsTools/Candidate/interface/Candidate.h"
#include "PhysicsTools/Candidate/interface/daughter_const_iterator.h"

namespace phystools {

  template<typename O>
  struct Overlap {
    explicit Overlap( const O & o ) : overlap( o ) { }
    bool operator()( const Candidate &, const Candidate & ) const;
    
  private: 
    O overlap;
  };
  
  
  template<typename O>
  bool Overlap<O>::operator()( const Candidate & c1, const Candidate & c2 ) const {
    typedef Candidate::const_iterator iterator;
    if( c1.numberOfDaughters() == 0 ) {
      if ( c2.numberOfDaughters() == 0 ) {
	return overlap( c1, c2 );
      }
      for( iterator i2 = c2.begin(); i2 != c2.end(); ++ i2 ) {
	if( operator()( c1, * i2 ) ) { 
	  return true;
	}
      }
      return false;
    }
    for( iterator i1 = c1.begin(); i1 != c1.end(); ++ i1 ) {
      if( operator()( * i1, c2 ) ) { 
	return true;
      }
    }
    return false;
  }
  
}

#endif
