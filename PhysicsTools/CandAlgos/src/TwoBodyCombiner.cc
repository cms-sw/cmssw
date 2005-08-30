// $Id: TwoBodyCombiner.cc,v 1.1 2005/07/29 07:22:52 llista Exp $

#include "PhysicsTools/CandAlgos/src/TwoBodyCombiner.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
using namespace phystools;
using namespace edm;
#include <iostream>

TwoBodyCombiner::TwoBodyCombiner( const ParameterSet & p ) :
  mass2min( p.getParameter<double>( "massMin" ) ),
  mass2max( p.getParameter<double>( "massMax" ) ),
  checkCharge( p.getParameter<bool>( "checkCharge" ) ),
  charge( 0 ),
  overlap( TrackOverlapChecker() ) {
  produces<Candidates>();
  if ( checkCharge ) charge = p.getParameter<int>( "charge" );
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
 
Candidate * TwoBodyCombiner::combine( const Candidate & c1, const Candidate & c2 ) {
  CompositeCandidate * cmp = new CompositeCandidate;
  cmp->addDaughter( c1 );
  cmp->addDaughter( c2 );
  cmp->setup( addp4 );
  return cmp;
}
