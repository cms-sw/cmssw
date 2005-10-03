// $Id: TwoBodyCombiner.cc,v 1.3 2005/10/01 22:16:59 llista Exp $
#include "PhysicsTools/CandUtils/interface/TwoBodyCombiner.h"
using namespace phystools;

TwoBodyCombiner::TwoBodyCombiner( double massMin, double massMax, 
				  bool ck, int q ) :
  mass2min( massMin ), mass2max( massMax ),
  checkCharge( ck ), charge( 0 ),
  overlap() {
  if ( checkCharge ) charge = q;
  mass2min *= mass2min;
  mass2max *= mass2max;
}
	 
bool TwoBodyCombiner::select( const Candidate & c1, const Candidate & c2 ) const {
  if ( checkCharge ) {
    int q = c1.charge() + c2.charge();
    if ( q != charge ) return false;
  }
  double mass2 = ( c1.p4() + c2.p4() ).mag2();
  if ( mass2min > mass2 || mass2 > mass2max ) return false;
  if ( overlap( c1, c2 ) ) return false;
  return true;
}

// the tollowing could be risky if an exception is thrown...
Candidate * TwoBodyCombiner::combine( const Candidate & c1, const Candidate & c2 ) {
  CompositeCandidate * cmp( new CompositeCandidate );
  cmp->addDaughter( c1 );
  cmp->addDaughter( c2 );
  cmp->setup( addp4 );
  return cmp;
}
