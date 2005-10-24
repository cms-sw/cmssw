// $Id: TwoBodyProducer.cc,v 1.2 2005/10/03 10:12:11 llista Exp $
#include "PhysicsTools/CandAlgos/src/TwoBodyProducer.h"
#include "PhysicsTools/CandUtils/interface/MassWindowSelector.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
using namespace aod;
using namespace edm;
using namespace std;

TwoBodyProducer::TwoBodyProducer( const ParameterSet & p ) :
  // bad? http://www.boost.org/libs/smart_ptr/shared_ptr.htm
  combiner( boost::shared_ptr<aod::Selector>( new MassWindowSelector( p.getParameter<double>( "massMin" ), 
								      p.getParameter<double>( "massMax" ) ) ),
	    p.getParameter<bool>( "checkCharge" ),
	    p.getParameter<bool>( "checkCharge" ) ? p.getParameter<int>( "charge" ) : 0 ),
  source1( p.getParameter<string>( "src1" ) ),
  source2( p.getParameter<string>( "src2" ) ) {
  produces<TwoBodyCombiner::Candidates>();
}

TwoBodyProducer::~TwoBodyProducer() {
}

void TwoBodyProducer::produce( Event& evt, const EventSetup& ) {
  Handle<TwoBodyCombiner::Candidates> cands1, cands2;
  evt.getByLabel( source1, cands1 );
  evt.getByLabel( source2, cands2 );
  evt.put( combiner.combine( & * cands1, & * cands2 ) );
}
