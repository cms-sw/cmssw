#include "RecoParticleFlow/PFProducer/plugins/GsfElectronLinker.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoParticleFlow/PFProducer/interface/GsfElectronEqual.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


GsfElectronLinker::GsfElectronLinker(const edm::ParameterSet & iConfig) {
  inputTagPFCandidates_ 
    = iConfig.getParameter<edm::InputTag>("PFCandidate");
  inputTagGsfElectrons_
    = iConfig.getParameter<edm::InputTag>("GsfElectrons");
  nameOutputPF_ 
    = iConfig.getParameter<std::string>("OutputPF");
  
  produces<reco::PFCandidateCollection>(nameOutputPF_);
  produces<edm::ValueMap<reco::PFCandidateRef> > (nameOutputPF_);
}

GsfElectronLinker::~GsfElectronLinker() {;}

void GsfElectronLinker::beginRun(edm::Run& run,const edm::EventSetup & es) {;}

void GsfElectronLinker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::auto_ptr<reco::PFCandidateCollection>
    pfCandidates_p(new reco::PFCandidateCollection);

  std::auto_ptr<edm::ValueMap<reco::PFCandidateRef> > 
    pfMap_p(new edm::ValueMap<reco::PFCandidateRef>());
  edm::ValueMap<reco::PFCandidateRef>::Filler pfMapFiller(*pfMap_p);

  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  bool status=fetchCandidateCollection(pfCandidates, 
				       inputTagPFCandidates_, 
				       iEvent );

  edm::Handle<reco::GsfElectronCollection> gsfElectrons;
  status=fetchGsfElectronCollection(gsfElectrons,
				    inputTagGsfElectrons_,
				    iEvent );

  unsigned ncand=(status)?pfCandidates->size():0;

  for( unsigned i=0; i<ncand; ++i ) {
    edm::Ptr<reco::PFCandidate> candPtr(pfCandidates,i);
    reco::PFCandidate cand = (*pfCandidates)[i];    
    cand.setSourceCandidatePtr(candPtr);
    
    // if not an electron or not GsfTrackRef
    if( (cand.particleId()!=reco::PFCandidate::e) ) {
      pfCandidates_p->push_back(cand);
      continue; // Watch out ! Continue
    }
    
    // if it is an electron. Find the GsfElectron with the same GsfTrack
    const reco::GsfTrackRef & gsfTrackRef(cand.gsfTrackRef());
    GsfElectronEqual myEqual(gsfTrackRef);
    std::vector<reco::GsfElectron>::const_iterator itcheck=find_if(gsfElectrons->begin(),gsfElectrons->end(),myEqual);
    if(itcheck==gsfElectrons->end()) {
      std::ostringstream err;
      err << " Problem in GsfElectronLinker: no GsfElectron " << std::endl;
      edm::LogError("GsfElectronLinker") << err.str();
      continue; // Watch out ! Continue
    } 
    reco::GsfElectronRef electronRef(gsfElectrons,itcheck-gsfElectrons->begin());
    cand.setGsfElectronRef(electronRef);
    electronCandidateMap_[electronRef]=i;
    pfCandidates_p->push_back(cand);
  }  

  const edm::OrphanHandle<reco::PFCandidateCollection> pfCandidateRefProd = 
    iEvent.put(pfCandidates_p,nameOutputPF_);

  fillValueMap(gsfElectrons,pfCandidateRefProd,pfMapFiller);
  iEvent.put(pfMap_p,nameOutputPF_);
}


bool GsfElectronLinker::fetchCandidateCollection(edm::Handle<reco::PFCandidateCollection>& c, 
						    const edm::InputTag& tag, 
						    const edm::Event& iEvent) const {  
  bool found = iEvent.getByLabel(tag, c);
  
  if(!found )
    {
      std::ostringstream  err;
      err<<" cannot get PFCandidates: "
	 <<tag<<std::endl;
      edm::LogError("GsfElectronLinker")<<err.str();
    }
  return found;
  
}

bool GsfElectronLinker::fetchGsfElectronCollection(edm::Handle<reco::GsfElectronCollection>& c, 
						const edm::InputTag& tag, 
						const edm::Event& iEvent) const {
  bool found = iEvent.getByLabel(tag, c);
  
  if(!found )
    {
      std::ostringstream  err;
      err<<" cannot get GsfElectrons: "
	 <<tag<<std::endl;
      edm::LogError("GsfElectronLinker")<<err.str();
    }
  return found;

}

void GsfElectronLinker::fillValueMap(edm::Handle<reco::GsfElectronCollection>& electrons,
				  const edm::OrphanHandle<reco::PFCandidateCollection> & pfHandle,
				  edm::ValueMap<reco::PFCandidateRef>::Filler & filler) const {
  unsigned nElectrons=electrons->size();
  std::vector<reco::PFCandidateRef> values;
  for(unsigned ielec=0;ielec<nElectrons;++ielec) {
    reco::GsfElectronRef gsfElecRef(electrons,ielec);
    std::map<reco::GsfElectronRef,unsigned>::const_iterator itcheck=electronCandidateMap_.find(gsfElecRef);
    if(itcheck==electronCandidateMap_.end()) {
      values.push_back(reco::PFCandidateRef());
    } else {
      values.push_back(reco::PFCandidateRef(pfHandle,itcheck->second));
    }
  }
  filler.insert(electrons,values.begin(),values.end());
}
