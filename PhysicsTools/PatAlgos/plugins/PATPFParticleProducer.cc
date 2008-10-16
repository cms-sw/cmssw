//
// $Id: PATPFParticleProducer.cc,v 1.12 2008/07/21 17:18:38 gpetrucc Exp $
//

#include "PhysicsTools/PatAlgos/plugins/PATPFParticleProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "DataFormats/Common/interface/Association.h"

#include "PhysicsTools/PatUtils/interface/ObjectResolutionCalc.h"

#include "TMath.h"

#include <vector>
#include <memory>


using namespace pat;


PATPFParticleProducer::PATPFParticleProducer(const edm::ParameterSet & iConfig) {
  // general configurables
  pfCandidateSrc_ = iConfig.getParameter<edm::InputTag>( "pfCandidateSource" );
 
  // MC matching configurables
  addGenMatch_   = iConfig.getParameter<bool>         ( "addGenMatch" );
  embedGenMatch_ = iConfig.getParameter<bool>         ( "embedGenMatch" );
  // genMatchSrc_   = iConfig.getParameter<edm::InputTag>( "genParticleMatch" );


  // produces vector of muons
  produces<std::vector<PFParticle> >();

}


PATPFParticleProducer::~PATPFParticleProducer() {
}


void PATPFParticleProducer::produce(edm::Event & iEvent, 
				    const edm::EventSetup & iSetup) {

  // Get the collection of PFCandidates from the event
  edm::Handle<edm::View<PFParticleType> > pfCandidates;

  fetchCandidateCollection(pfCandidates, 
			   pfCandidateSrc_, 
			   iEvent );

  // prepare the MC matching
  edm::Handle<edm::Association<reco::GenParticleCollection> > genMatch;
  if (addGenMatch_) iEvent.getByLabel(genMatchSrc_, genMatch);

  // loop over PFCandidates
  std::vector<PFParticle> * patPFParticles = new std::vector<PFParticle>();
  for (edm::View<PFParticleType>::const_iterator 
	 itPFParticle = pfCandidates->begin(); 
       itPFParticle != pfCandidates->end(); 
       ++itPFParticle) {

    // construct the PFParticle from the ref -> save ref to original object
    unsigned int idx = itPFParticle - pfCandidates->begin();
    edm::RefToBase<PFParticleType> pfCandidatesRef = pfCandidates->refAt(idx);

    PFParticle aPFParticle(pfCandidatesRef);

    // store the match to the generated final state pfCandidates
//     if (addGenMatch_) {
//       reco::GenParticleRef genPFParticle = (*genMatch)[pfCandidatesRef];
//       if (genPFParticle.isNonnull() && genPFParticle.isAvailable() ) {
//         aPFParticle.setGenLepton(genPFParticle, embedGenMatch_);
//       } // leave empty if no match found
//     }


    // add sel to selected
    patPFParticles->push_back(aPFParticle);
  }

  // sort pfCandidates in pt
  std::sort(patPFParticles->begin(), patPFParticles->end(), pTComparator_);

  // put genEvt object in Event
  std::auto_ptr<std::vector<PFParticle> > ptr(patPFParticles);
  iEvent.put(ptr);

}

void 
PATPFParticleProducer::fetchCandidateCollection( edm::Handle< edm::View<PFParticleType> >& c, 
						 const edm::InputTag& tag, 
						 const edm::Event& iEvent) const {
  
  bool found = iEvent.getByLabel(tag, c);
  
  if(!found ) {
    std::ostringstream  err;
    err<<" cannot get PFCandidates: "
       <<tag<<std::endl;
    edm::LogError("PFCandidates")<<err.str();
    throw cms::Exception( "MissingProduct", err.str());
  }
  
}



#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATPFParticleProducer);
