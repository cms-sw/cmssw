// $Id: TwoBodyProducer.cc,v 1.1 2005/10/24 11:40:59 llista Exp $
#include "PhysicsTools/CandAlgos/src/TwoBodyCombiner.h"
#include "PhysicsTools/CandUtils/interface/MassWindowSelector.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
using namespace aod;
using namespace edm;
using namespace std;

candmodules::TwoBodyCombiner::TwoBodyCombiner( const ParameterSet & p ) :
  // bad? http://www.boost.org/libs/smart_ptr/shared_ptr.htm
  combiner( boost::shared_ptr<aod::Candidate::selector>( 
    new MassWindowSelector( p.getParameter<double>( "massMin" ), 
			    p.getParameter<double>( "massMax" ) ) ),
    p.getParameter<bool>( "checkCharge" ),
    p.getParameter<bool>( "checkCharge" ) ? p.getParameter<int>( "charge" ) : 0 ),
  source1( p.getParameter<string>( "src1" ) ),
  source2( p.getParameter<string>( "src2" ) ) {
  produces<Candidates>();
}

candmodules::TwoBodyCombiner::~TwoBodyCombiner() {
}

void candmodules::TwoBodyCombiner::produce( Event& evt, const EventSetup& ) {
  Handle<Candidates> cands1, cands2;
  evt.getByLabel( source1, cands1 );
  evt.getByLabel( source2, cands2 );
  evt.put( combiner.combine( & * cands1, & * cands2 ) );
}
