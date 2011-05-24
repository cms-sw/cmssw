#include "RecoParticleFlow/PFProducer/plugins/EgammaPFLinker.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoParticleFlow/PFProducer/interface/GsfElectronEqual.h"
#include "RecoParticleFlow/PFProducer/interface/PhotonEqual.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


EgammaPFLinker::EgammaPFLinker(const edm::ParameterSet & iConfig) {
  inputTagPFCandidates_ 
    = iConfig.getParameter<edm::InputTag>("PFCandidate");
  inputTagGsfElectrons_
    = iConfig.getParameter<edm::InputTag>("GsfElectrons");
  inputTagPhotons_
    = iConfig.getParameter<edm::InputTag>("Photons");
  nameOutputPF_ 
    = iConfig.getParameter<std::string>("OutputPF");
  
  nameOutputElectronsPF_ 
    = iConfig.getParameter<std::string>("ValueMapElectrons");

  nameOutputPhotonsPF_ 
    = iConfig.getParameter<std::string>("ValueMapPhotons");

  producePFCandidates_  
    = iConfig.getParameter<bool>("ProducePFCandidates");
  
  if(producePFCandidates_) {
    produces<reco::PFCandidateCollection>(nameOutputPF_);
  }
  produces<edm::ValueMap<reco::PFCandidatePtr> > (nameOutputElectronsPF_);
  produces<edm::ValueMap<reco::PFCandidatePtr> > (nameOutputPhotonsPF_);
}

EgammaPFLinker::~EgammaPFLinker() {;}

void EgammaPFLinker::beginRun(edm::Run& run,const edm::EventSetup & es) {;}

void EgammaPFLinker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  electronCandidateMap_.clear();
  std::auto_ptr<reco::PFCandidateCollection>
    pfCandidates_p(new reco::PFCandidateCollection);

  std::auto_ptr<edm::ValueMap<reco::PFCandidatePtr> > 
    pfMapGsfElectrons_p(new edm::ValueMap<reco::PFCandidatePtr>());
  edm::ValueMap<reco::PFCandidatePtr>::Filler pfMapGsfElectronFiller(*pfMapGsfElectrons_p);

  std::auto_ptr<edm::ValueMap<reco::PFCandidatePtr> > 
    pfMapPhotons_p(new edm::ValueMap<reco::PFCandidatePtr>());
  edm::ValueMap<reco::PFCandidatePtr>::Filler pfMapPhotonFiller(*pfMapPhotons_p);

  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  bool status=fetchCandidateCollection(pfCandidates, 
				       inputTagPFCandidates_, 
				       iEvent );

  edm::Handle<reco::GsfElectronCollection> gsfElectrons;
  status=fetchGsfElectronCollection(gsfElectrons,
				    inputTagGsfElectrons_,
				    iEvent );

  edm::Handle<reco::PhotonCollection> photons;
  status=fetchPhotonCollection(photons,
			       inputTagPhotons_,
			       iEvent );

  unsigned ncand=(status)?pfCandidates->size():0;

  for( unsigned i=0; i<ncand; ++i ) {
    edm::Ptr<reco::PFCandidate> candPtr(pfCandidates,i);
    reco::PFCandidate cand(candPtr);
    
    // if not an electron or a photon with mva_nothing_gamma>0 
    if(! (cand.particleId()==reco::PFCandidate::e) || ((cand.particleId()==reco::PFCandidate::gamma)&&(cand.mva_nothing_gamma()>0.))) {
      pfCandidates_p->push_back(cand);
      // watch out
      continue;
    }

    // if it is an electron. Find the GsfElectron with the same GsfTrack
    if (cand.particleId()==reco::PFCandidate::e) {
      const reco::GsfTrackRef & gsfTrackRef(cand.gsfTrackRef());
      GsfElectronEqual myEqual(gsfTrackRef);
      std::vector<reco::GsfElectron>::const_iterator itcheck=find_if(gsfElectrons->begin(),gsfElectrons->end(),myEqual);
      if(itcheck==gsfElectrons->end()) {
	std::ostringstream err;
	err << " Problem in EgammaPFLinker: no GsfElectron " << std::endl;
	edm::LogError("EgammaPFLinker") << err.str();
	continue; // Watch out ! Continue
      } 
      reco::GsfElectronRef electronRef(gsfElectrons,itcheck-gsfElectrons->begin());
      cand.setGsfElectronRef(electronRef);
      cand.setSuperClusterRef(electronRef->superCluster());
      electronCandidateMap_[electronRef]=i;
      pfCandidates_p->push_back(cand);
    }  
  
    // if it is a photon, find the one with the same PF super-cluster
    if (cand.particleId()==reco::PFCandidate::gamma && cand.mva_nothing_gamma()>0.) {
      const reco::SuperClusterRef & scRef(cand.superClusterRef());
      PhotonEqual myEqual(scRef);
      std::vector<reco::Photon>::const_iterator itcheck=find_if(photons->begin(),photons->end(),myEqual);
      if(itcheck==photons->end()) {
	std::ostringstream err;
	err << " Problem in EgammaPFLinker: no Photon " << std::endl;
	edm::LogError("EgammaPFLinker") << err.str();
	continue; // Watch out ! Continue
      } 
      reco::PhotonRef photonRef(photons,itcheck-photons->begin());
      cand.setPhotonRef(photonRef);
      cand.setSuperClusterRef(photonRef->superCluster());
      photonCandidateMap_[photonRef]=i;
      pfCandidates_p->push_back(cand);
    }
    
  }
  // save the PFCandidates and get a valid handle

  const edm::OrphanHandle<reco::PFCandidateCollection> pfCandidateRefProd = (producePFCandidates_) ? iEvent.put(pfCandidates_p,nameOutputPF_) :
    edm::OrphanHandle<reco::PFCandidateCollection>();

  
  // now make the valuemaps
  fillValueMap(gsfElectrons,pfCandidateRefProd,pfCandidates,pfMapGsfElectronFiller);  
  fillValueMap(photons,pfCandidateRefProd,pfCandidates,pfMapPhotonFiller);  
  
  iEvent.put(pfMapGsfElectrons_p,nameOutputElectronsPF_);
  iEvent.put(pfMapPhotons_p,nameOutputPhotonsPF_);
}


bool EgammaPFLinker::fetchCandidateCollection(edm::Handle<reco::PFCandidateCollection>& c, 
						    const edm::InputTag& tag, 
						    const edm::Event& iEvent) const {  
  bool found = iEvent.getByLabel(tag, c);
  
  if(!found )
    {
      std::ostringstream  err;
      err<<" cannot get PFCandidates: "
	 <<tag<<std::endl;
      edm::LogError("EgammaPFLinker")<<err.str();
    }
  return found;
  
}

bool EgammaPFLinker::fetchGsfElectronCollection(edm::Handle<reco::GsfElectronCollection>& c, 
						   const edm::InputTag& tag, 
						   const edm::Event& iEvent) const {
  bool found = iEvent.getByLabel(tag, c);
  
  if(!found )
    {
      std::ostringstream  err;
      err<<" cannot get GsfElectrons: "
	 <<tag<<std::endl;
      edm::LogError("EgammaPFLinker")<<err.str();
    }
  return found;

}

bool EgammaPFLinker::fetchPhotonCollection(edm::Handle<reco::PhotonCollection>& c, 
					      const edm::InputTag& tag, 
					      const edm::Event& iEvent) const {
  bool found = iEvent.getByLabel(tag, c);
  
  if(!found )
    {
      std::ostringstream  err;
      err<<" cannot get Photons: "
	 <<tag<<std::endl;
      edm::LogError("EgammaPFLinker")<<err.str();
    }
  return found;

}


void EgammaPFLinker::fillValueMap(edm::Handle<reco::GsfElectronCollection>& electrons,
				  const edm::OrphanHandle<reco::PFCandidateCollection> & pfOrphanHandle,
				  const edm::Handle<reco::PFCandidateCollection> & pfHandle,
				  edm::ValueMap<reco::PFCandidatePtr>::Filler & filler) const {
  unsigned nElectrons=electrons->size();
  std::vector<reco::PFCandidatePtr> values;
  for(unsigned ielec=0;ielec<nElectrons;++ielec) {
    reco::GsfElectronRef gsfElecRef(electrons,ielec);
    std::map<reco::GsfElectronRef,unsigned>::const_iterator itcheck=electronCandidateMap_.find(gsfElecRef);
    if(itcheck==electronCandidateMap_.end()) {
      values.push_back(reco::PFCandidatePtr());
    } else {
      if(producePFCandidates_) {
	// points to the new collection
	values.push_back(reco::PFCandidatePtr(pfOrphanHandle,itcheck->second));
      }
      else { 
	// points to the old one
	values.push_back(reco::PFCandidatePtr(pfHandle,itcheck->second));
      }
    }
  }
  filler.insert(electrons,values.begin(),values.end());
}

void EgammaPFLinker::fillValueMap(edm::Handle<reco::PhotonCollection>& photons,
				     const edm::OrphanHandle<reco::PFCandidateCollection> & pfOrphanHandle,
				     const edm::Handle<reco::PFCandidateCollection> & pfHandle,
				     edm::ValueMap<reco::PFCandidatePtr>::Filler & filler) const {
  unsigned nPhotons=photons->size();
  std::vector<reco::PFCandidatePtr> values;
  for(unsigned iphot=0;iphot<nPhotons;++iphot) {
    reco::PhotonRef photonRef(photons,iphot);
    std::map<reco::PhotonRef,unsigned>::const_iterator itcheck=photonCandidateMap_.find(photonRef);
    if(itcheck==photonCandidateMap_.end()) {
      values.push_back(reco::PFCandidatePtr());
    } else {
      if(producePFCandidates_) {
	// points to the new collection
	values.push_back(reco::PFCandidatePtr(pfOrphanHandle,itcheck->second));
      }
      else {
	// points to the old collection
	values.push_back(reco::PFCandidatePtr(pfHandle,itcheck->second));
      }
    }
  }
  filler.insert(photons,values.begin(),values.end());
}
