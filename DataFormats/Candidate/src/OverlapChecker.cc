// $Id: OverlapChecker.cc,v 1.1 2006/02/28 10:43:30 llista Exp $
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

