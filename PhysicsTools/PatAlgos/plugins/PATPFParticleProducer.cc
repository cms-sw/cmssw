//
//

#include "PhysicsTools/PatAlgos/plugins/PATPFParticleProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "DataFormats/Common/interface/Association.h"

#include "TMath.h"

#include <vector>
#include <memory>


using namespace pat;


PATPFParticleProducer::PATPFParticleProducer(const edm::ParameterSet & iConfig) :
    userDataHelper_ ( iConfig.getParameter<edm::ParameterSet>("userData"), consumesCollector() )
{
  // general configurables
  pfCandidateToken_ = consumes<edm::View<reco::PFCandidate> >(iConfig.getParameter<edm::InputTag>( "pfCandidateSource" ));

  // MC matching configurables
  addGenMatch_   = iConfig.getParameter<bool> ( "addGenMatch" );
  if (addGenMatch_) {
    embedGenMatch_ = iConfig.getParameter<bool>( "embedGenMatch" );
    if (iConfig.existsAs<edm::InputTag>("genParticleMatch")) {
      genMatchTokens_.push_back(consumes<edm::Association<reco::GenParticleCollection> >(iConfig.getParameter<edm::InputTag>( "genParticleMatch" )));
    }
    else {
      genMatchTokens_ = edm::vector_transform(iConfig.getParameter<std::vector<edm::InputTag> >( "genParticleMatch" ), [this](edm::InputTag const & tag){return consumes<edm::Association<reco::GenParticleCollection> >(tag);});
    }
  }

  // Efficiency configurables
  addEfficiencies_ = iConfig.getParameter<bool>("addEfficiencies");
  if (addEfficiencies_) {
     efficiencyLoader_ = pat::helper::EfficiencyLoader(iConfig.getParameter<edm::ParameterSet>("efficiencies"), consumesCollector());
  }

  // Resolution configurables
  addResolutions_ = iConfig.getParameter<bool>("addResolutions");
  if (addResolutions_) {
     resolutionLoader_ = pat::helper::KinResolutionsLoader(iConfig.getParameter<edm::ParameterSet>("resolutions"));
  }

  // Check to see if the user wants to add user data
  useUserData_ = false;
  if ( iConfig.exists("userData") ) {
    useUserData_ = true;
  }

  // produces vector of muons
  produces<std::vector<PFParticle> >();

}


PATPFParticleProducer::~PATPFParticleProducer() {
}


void PATPFParticleProducer::produce(edm::Event & iEvent,
				    const edm::EventSetup & iSetup) {

  // Get the collection of PFCandidates from the event
  edm::Handle<edm::View<reco::PFCandidate> > pfCandidates;
  iEvent.getByToken(pfCandidateToken_, pfCandidates);

  // prepare the MC matching
  std::vector<edm::Handle<edm::Association<reco::GenParticleCollection> > > genMatches(genMatchTokens_.size());
  if (addGenMatch_) {
    for (size_t j = 0, nd = genMatchTokens_.size(); j < nd; ++j) {
      iEvent.getByToken(genMatchTokens_[j], genMatches[j]);
    }
  }

  if (efficiencyLoader_.enabled()) efficiencyLoader_.newEvent(iEvent);
  if (resolutionLoader_.enabled()) resolutionLoader_.newEvent(iEvent, iSetup);

  // loop over PFCandidates
  std::vector<PFParticle> * patPFParticles = new std::vector<PFParticle>();
  for (edm::View<reco::PFCandidate>::const_iterator
	 itPFParticle = pfCandidates->begin();
       itPFParticle != pfCandidates->end();
       ++itPFParticle) {

    // construct the PFParticle from the ref -> save ref to original object
    unsigned int idx = itPFParticle - pfCandidates->begin();
    edm::RefToBase<reco::PFCandidate> pfCandidatesRef = pfCandidates->refAt(idx);

    PFParticle aPFParticle(pfCandidatesRef);

    if (addGenMatch_) {
      for(size_t i = 0, n = genMatches.size(); i < n; ++i) {
          reco::GenParticleRef genPFParticle = (*genMatches[i])[pfCandidatesRef];
          aPFParticle.addGenParticleRef(genPFParticle);
      }
      if (embedGenMatch_) aPFParticle.embedGenParticle();
    }

    if (efficiencyLoader_.enabled()) {
        efficiencyLoader_.setEfficiencies( aPFParticle, pfCandidatesRef );
    }

    if (resolutionLoader_.enabled()) {
        resolutionLoader_.setResolutions(aPFParticle);
    }

    if ( useUserData_ ) {
        userDataHelper_.add( aPFParticle, iEvent, iSetup );
    }


    // add sel to selected
    patPFParticles->push_back(aPFParticle);
  }

  // sort pfCandidates in pt
  std::sort(patPFParticles->begin(), patPFParticles->end(), pTComparator_);

  // put genEvt object in Event
  std::unique_ptr<std::vector<PFParticle> > ptr(patPFParticles);
  iEvent.put(std::move(ptr));

}



#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATPFParticleProducer);
