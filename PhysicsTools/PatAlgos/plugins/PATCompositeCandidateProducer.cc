//
// $Id: PATCompositeCandidateProducer.cc,v 1.8 2008/11/04 15:42:03 gpetrucc Exp $
//

#include "PhysicsTools/PatAlgos/plugins/PATCompositeCandidateProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/View.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "PhysicsTools/Utilities/interface/StringObjectFunction.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <memory>


#include <iostream>

using namespace pat;
using namespace std;
using namespace edm;

PATCompositeCandidateProducer::PATCompositeCandidateProducer(const ParameterSet & iConfig) :
  userDataHelper_( iConfig.getParameter<edm::ParameterSet>("userData") )
{
  // initialize the configurables
  src_ = iConfig.getParameter<InputTag>( "src" );

  useUserData_ = false;
  if ( iConfig.exists("userData") ) {
    useUserData_ = true;
  }

  // produces vector of particles
  produces<vector<pat::CompositeCandidate> >();

}

PATCompositeCandidateProducer::~PATCompositeCandidateProducer() {
}

void PATCompositeCandidateProducer::produce(Event & iEvent, const EventSetup & iSetup) {
  // Get the vector of CompositeCandidate's from the event
  Handle<View<reco::CompositeCandidate> > cands;
  iEvent.getByLabel(src_, cands);

  auto_ptr<vector<pat::CompositeCandidate> > myCompositeCandidates ( new vector<pat::CompositeCandidate>() );

  if ( cands.isValid() ) {

    View<reco::CompositeCandidate>::const_iterator ibegin = cands->begin(),
      iend = cands->end(), i = ibegin;
    for ( ; i != iend; ++i ) {

      pat::CompositeCandidate cand(*i);
      
      if ( useUserData_ ) {
	userDataHelper_.add( cand, iEvent, iSetup );
      }

      myCompositeCandidates->push_back( cand );
    }

  }// end if the two handles are valid

  iEvent.put(myCompositeCandidates);

}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATCompositeCandidateProducer);
