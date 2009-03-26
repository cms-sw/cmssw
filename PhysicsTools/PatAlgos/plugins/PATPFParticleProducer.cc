//
// $Id: PATPFParticleProducer.cc,v 1.3 2008/11/28 22:05:56 lowette Exp $
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


PATPFParticleProducer::PATPFParticleProducer(const edm::ParameterSet & iConfig) {
  // general configurables
  pfCandidateSrc_ = iConfig.getParameter<edm::InputTag>( "pfCandidateSource" );
 
  // MC matching configurables
  addGenMatch_   = iConfig.getParameter<bool>         ( "addGenMatch" );
  if (addGenMatch_) {
      embedGenMatch_ = iConfig.getParameter<bool>         ( "embedGenMatch" );
      if (iConfig.existsAs<edm::InputTag>("genParticleMatch")) {
          genMatchSrc_.push_back(iConfig.getParameter<edm::InputTag>( "genParticleMatch" ));
      } else {
          genMatchSrc_ = iConfig.getParameter<std::vector<edm::InputTag> >( "genParticleMatch" );
      }
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

  fetchCandidateCollection(pfCandidates, 
			   pfCandidateSrc_, 
			   iEvent );

  // prepare the MC matching
  std::vector<edm::Handle<edm::Association<reco::GenParticleCollection> > > genMatches(genMatchSrc_.size());
  if (addGenMatch_) {
        for (size_t j = 0, nd = genMatchSrc_.size(); j < nd; ++j) {
            iEvent.getByLabel(genMatchSrc_[j], genMatches[j]);
        }
  }

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
PATPFParticleProducer::fetchCandidateCollection( edm::Handle<edm::View<reco::PFCandidate> >& c, 
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
