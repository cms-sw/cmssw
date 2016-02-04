// $Id: OverlapChecker.cc,v 1.3 2007/09/27 13:51:55 llista Exp $
#include "DataFormats/Candidate/interface/OverlapChecker.h"  
#include "DataFormats/Candidate/interface/Candidate.h"
using namespace reco;
  
bool OverlapChecker::operator()( const Candidate & c1, const Candidate & c2 ) const {
  typedef Candidate::const_iterator iterator;
  if( c1.numberOfDaughters() == 0 ) {
    if ( c2.numberOfDaughters() == 0 ) {
      if( c2.hasMasterClone() )
	return c1.overlap( *(c2.masterClone()) );
      else
	return c1.overlap( c2 );
    }
    iterator b2 = c2.begin(), e2 = c2.end();
    for( iterator i2 = b2; i2 != e2; ++ i2 ) {
      if( operator()( c1, * i2 ) ) { 
	return true;
      }
    }
    return false;
  }
  iterator b1 = c1.begin(), e1 = c1.end();
  for( iterator i1 = b1; i1 != e1; ++ i1 ) {
    if( operator()( * i1, c2 ) ) { 
      return true;
    }
  }
  return false;
}

