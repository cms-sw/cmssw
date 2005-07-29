#include "PhysicsTools/CandAlgos/src/TwoSameBodyCombiner.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
using namespace phystools;
using namespace edm;
using namespace std;

TwoSameBodyCombiner::TwoSameBodyCombiner( const ParameterSet & p ) :
  TwoBodyCombiner( p ),
  source( p.getParameter<string>( "src" ) ) {
}

void TwoSameBodyCombiner::produce( Event& evt, const EventSetup& ) {
  Handle<Candidates> cands;
  try {
    evt.getByLabel( source, cands );
  } catch ( exception e ) {
    cerr << "Error: can't get collection " << source << endl;
    return;
  }

  auto_ptr<Candidates> comps( new Candidates );
  const int n = cands->size();
  for( int i1 = 0; i1 < n; ++ i1 ) {
    const Candidate & c1 = * (*cands)[ i1 ];
    for ( int i2 = i1 + 1; i2 < n; ++ i2 ) {
      const Candidate & c2 = * (*cands)[ i2 ];
      if ( select( c1, c2 ) ) {
	comps->push_back( combine( c1, c2 ) );
      }
    }
  }
  evt.put( comps );
}
