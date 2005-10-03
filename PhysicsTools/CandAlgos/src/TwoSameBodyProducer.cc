#include "PhysicsTools/CandAlgos/src/TwoSameBodyProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
using namespace phystools;
using namespace edm;
using namespace std;

TwoSameBodyProducer::TwoSameBodyProducer( const ParameterSet & p ) :
  combiner( p.getParameter<double>( "massMin" ), 
	    p.getParameter<double>( "massMax" ),
	    p.getParameter<bool>( "checkCharge" ),
	    p.getParameter<bool>( "checkCharge" ) ? p.getParameter<int>( "charge" ) : 0 ),
  source( p.getParameter<string>( "src" ) ) {
  produces<TwoBodyCombiner::Candidates>();
}

void TwoSameBodyProducer::produce( Event& evt, const EventSetup& ) {
  Handle<TwoBodyCombiner::Candidates> cands;
  try {
    evt.getByLabel( source, cands );
  } catch ( exception e ) {
    cerr << "Error: can't get collection " << source << endl;
    return;
  }

  evt.put( combiner.combine( * cands ) );
}
