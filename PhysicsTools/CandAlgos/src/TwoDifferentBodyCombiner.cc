// $Id$

#include "PhysicsTools/CandAlgos/src/TwoDifferentBodyCombiner.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
using namespace phystools;
using namespace edm;
using namespace std;

TwoDifferentBodyCombiner::TwoDifferentBodyCombiner( const ParameterSet & p ) :
  TwoBodyCombiner( p ),
  source1( p.getParameter<string>( "src1" ) ),
  source2( p.getParameter<string>( "src2" ) ) {
}

void TwoDifferentBodyCombiner::produce( Event& evt, const EventSetup& ) {
  Handle<Candidates> cands1, cands2;
  try {
    evt.getByLabel( source1, cands1 );
  } catch ( exception e ) {
    cerr << "Error: can't get collection " << source1 << endl;
    return;
  }
  try {
    evt.getByLabel( source2, cands2 );
  } catch ( exception e ) {
    cerr << "Error: can't get collection " << source2 << endl;
    return;
  }

  auto_ptr<Candidates> comps( new Candidates );
  const int n1 = cands1->size(), n2 = cands2->size();
  for( int i1 = 0; i1 < n1; ++ i1 ) {
    const Candidate & c1 = * (*cands1)[ i1 ];
    for ( int i2 = 0; i2 < n2; ++ i2 ) {
      const Candidate & c2 = * (*cands2)[ i2 ];
      if ( select( c1, c2 ) ) {
	comps->push_back( combine( c1, c2 ) );
      }
    }
  }
  evt.put( comps );
}
