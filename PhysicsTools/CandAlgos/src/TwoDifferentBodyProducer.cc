// $Id: TwoDifferentBodyCombiner.cc,v 1.1 2005/07/29 07:22:52 llista Exp $
#include "PhysicsTools/CandAlgos/src/TwoDifferentBodyProducer.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
using namespace phystools;
using namespace edm;
using namespace std;

TwoDifferentBodyProducer::TwoDifferentBodyProducer( const ParameterSet & p ) :
  combiner( p.getParameter<double>( "massMin" ), 
	    p.getParameter<double>( "massMax" ),
	    p.getParameter<bool>( "checkCharge" ),
	    p.getParameter<bool>( "checkCharge" ) ? p.getParameter<int>( "charge" ) : 0 ),
  source1( p.getParameter<string>( "src1" ) ),
  source2( p.getParameter<string>( "src2" ) ) {
  produces<TwoBodyCombiner::Candidates>();
}

void TwoDifferentBodyProducer::produce( Event& evt, const EventSetup& ) {
  Handle<TwoBodyCombiner::Candidates> cands1, cands2;
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

  evt.put( combiner.combine( * cands1, * cands2 ) );
}
