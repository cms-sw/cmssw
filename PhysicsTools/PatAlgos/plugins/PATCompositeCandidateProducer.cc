//
//

#include "PhysicsTools/PatAlgos/plugins/PATCompositeCandidateProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/View.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <memory>


#include <iostream>

using namespace pat;
using namespace std;
using namespace edm;

PATCompositeCandidateProducer::PATCompositeCandidateProducer(const ParameterSet & iConfig) :
  srcToken_(consumes<edm::View<reco::CompositeCandidate> >(iConfig.getParameter<InputTag>( "src" ))),
  useUserData_(iConfig.exists("userData")),
  userDataHelper_( iConfig.getParameter<edm::ParameterSet>("userData"), consumesCollector() ),
  addEfficiencies_(iConfig.getParameter<bool>("addEfficiencies")),  
  addResolutions_(iConfig.getParameter<bool>("addResolutions"))
{
 
  // Efficiency configurables
  if (addEfficiencies_) {
     efficiencyLoader_ = pat::helper::EfficiencyLoader(iConfig.getParameter<edm::ParameterSet>("efficiencies"), consumesCollector());
  }

  // Resolution configurables
  if (addResolutions_) {
     resolutionLoader_ = pat::helper::KinResolutionsLoader(iConfig.getParameter<edm::ParameterSet>("resolutions"));
  }
  
  // produces vector of particles
  produces<vector<pat::CompositeCandidate> >();

}

PATCompositeCandidateProducer::~PATCompositeCandidateProducer() {
}

void PATCompositeCandidateProducer::produce(Event & iEvent, const EventSetup & iSetup) {
  // Get the vector of CompositeCandidate's from the event
  Handle<View<reco::CompositeCandidate> > cands;
  iEvent.getByToken(srcToken_, cands);

  if (efficiencyLoader_.enabled()) efficiencyLoader_.newEvent(iEvent);
  if (resolutionLoader_.enabled()) resolutionLoader_.newEvent(iEvent, iSetup);

  auto myCompositeCandidates = std::make_unique<vector<pat::CompositeCandidate> >();

  if ( cands.isValid() ) {

    View<reco::CompositeCandidate>::const_iterator ibegin = cands->begin(),
      iend = cands->end(), i = ibegin;
    for ( ; i != iend; ++i ) {

      pat::CompositeCandidate cand(*i);

      if ( useUserData_ ) {
	userDataHelper_.add( cand, iEvent, iSetup );
      }

      if (efficiencyLoader_.enabled()) efficiencyLoader_.setEfficiencies( cand, cands->refAt(i - cands->begin()) );
      if (resolutionLoader_.enabled()) resolutionLoader_.setResolutions(cand);

      myCompositeCandidates->push_back( std::move(cand) );
    }

  }// end if the two handles are valid

  iEvent.put(std::move(myCompositeCandidates));

}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATCompositeCandidateProducer);
