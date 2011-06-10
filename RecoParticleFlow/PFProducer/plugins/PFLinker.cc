#include "RecoParticleFlow/PFProducer/plugins/PFLinker.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/MuonReco/interface/MuonToMuonMap.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "RecoParticleFlow/PFProducer/interface/GsfElectronEqual.h"
#include "RecoParticleFlow/PFProducer/interface/PhotonEqual.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


PFLinker::PFLinker(const edm::ParameterSet & iConfig) {
  inputTagPFCandidates_ 
    = iConfig.getParameter<edm::InputTag>("PFCandidate");
  inputTagGsfElectrons_
    = iConfig.getParameter<edm::InputTag>("GsfElectrons");
  inputTagPhotons_
    = iConfig.getParameter<edm::InputTag>("Photons");
  inputTagMuons_
    = iConfig.getParameter<edm::InputTag>("Muons");
  
  nameOutputPF_ 
    = iConfig.getParameter<std::string>("OutputPF");
  
  nameOutputElectronsPF_ 
    = iConfig.getParameter<std::string>("ValueMapElectrons");

  nameOutputPhotonsPF_ 
    = iConfig.getParameter<std::string>("ValueMapPhotons");

  producePFCandidates_  
    = iConfig.getParameter<bool>("ProducePFCandidates");
  
  fillMuonRefs_
    = iConfig.getParameter<bool>("FillMuonRefs");


  if(producePFCandidates_) {
    produces<reco::PFCandidateCollection>(nameOutputPF_);
  }
  produces<edm::ValueMap<reco::PFCandidatePtr> > (nameOutputElectronsPF_);
  produces<edm::ValueMap<reco::PFCandidatePtr> > (nameOutputPhotonsPF_);
  if(fillMuonRefs_)  produces<edm::ValueMap<reco::PFCandidatePtr> > (inputTagMuons_.label());

}

PFLinker::~PFLinker() {;}

void PFLinker::beginRun(edm::Run& run,const edm::EventSetup & es) {;}

void PFLinker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
  std::auto_ptr<reco::PFCandidateCollection>
    pfCandidates_p(new reco::PFCandidateCollection);
  
  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  bool status=fetchCollection<reco::PFCandidateCollection>(pfCandidates, 
							   inputTagPFCandidates_, 
							   iEvent );
  
  edm::Handle<reco::GsfElectronCollection> gsfElectrons;
  status=fetchCollection<reco::GsfElectronCollection>(gsfElectrons,
						      inputTagGsfElectrons_,
						      iEvent );
  std::map<reco::GsfElectronRef,unsigned int> electronCandidateMap;  


  edm::Handle<reco::PhotonCollection> photons;
  status=fetchCollection<reco::PhotonCollection>(photons,
						 inputTagPhotons_,
						 iEvent );
  std::map<reco::PhotonRef,unsigned int> photonCandidateMap;


  edm::Handle<reco::MuonToMuonMap> muonMap;
  if(fillMuonRefs_)
    status=fetchCollection<reco::MuonToMuonMap>(muonMap,
						inputTagMuons_,
						iEvent);
  std::map<reco::MuonRef,unsigned int> muonCandidateMap;



  unsigned ncand=(status)?pfCandidates->size():0;

  for( unsigned i=0; i<ncand; ++i) {
    edm::Ptr<reco::PFCandidate> candPtr(pfCandidates,i);
    reco::PFCandidate cand(candPtr);
    
    bool isphoton   = cand.particleId() == reco::PFCandidate::gamma && cand.mva_nothing_gamma()>0.;
    bool iselectron = cand.particleId() == reco::PFCandidate::e;
    bool ismuon     = cand.particleId() == reco::PFCandidate::mu && fillMuonRefs_;

    // if not an electron or a photon or a muon just fill the PFCandidate collection
    if ( !(isphoton || iselectron || ismuon)){pfCandidates_p->push_back(cand); continue;}
    
    
    if (ismuon && fillMuonRefs_) {
      reco::MuonRef muRef = (*muonMap)[cand.muonRef()];
      cand.setMuonRef(muRef);
      muonCandidateMap[muRef] = i;
    }
     



    // if it is an electron. Find the GsfElectron with the same GsfTrack
    if (iselectron) {
      const reco::GsfTrackRef & gsfTrackRef(cand.gsfTrackRef());
      GsfElectronEqual myEqual(gsfTrackRef);
      std::vector<reco::GsfElectron>::const_iterator itcheck=find_if(gsfElectrons->begin(),gsfElectrons->end(),myEqual);
      if(itcheck==gsfElectrons->end()) {
	std::ostringstream err;
	err << " Problem in PFLinker: no GsfElectron " << std::endl;
	edm::LogError("PFLinker") << err.str();
	continue; // Watch out ! Continue
      } 
      reco::GsfElectronRef electronRef(gsfElectrons,itcheck-gsfElectrons->begin());
      cand.setGsfElectronRef(electronRef);
      cand.setSuperClusterRef(electronRef->superCluster());
      electronCandidateMap[electronRef]=i;
    }  
  
    // if it is a photon, find the one with the same PF super-cluster
    if (isphoton) {
      const reco::SuperClusterRef & scRef(cand.superClusterRef());
      PhotonEqual myEqual(scRef);
      std::vector<reco::Photon>::const_iterator itcheck=find_if(photons->begin(),photons->end(),myEqual);
      if(itcheck==photons->end()) {
	std::ostringstream err;
	err << " Problem in PFLinker: no Photon " << std::endl;
	edm::LogError("PFLinker") << err.str();
	continue; // Watch out ! Continue
      } 
      reco::PhotonRef photonRef(photons,itcheck-photons->begin());
      cand.setPhotonRef(photonRef);
      cand.setSuperClusterRef(photonRef->superCluster());
      photonCandidateMap[photonRef]=i;
    }

    pfCandidates_p->push_back(cand);
    
  }
  // save the PFCandidates and get a valid handle

  const edm::OrphanHandle<reco::PFCandidateCollection> pfCandidateRefProd = (producePFCandidates_) ? iEvent.put(pfCandidates_p,nameOutputPF_) :
    edm::OrphanHandle<reco::PFCandidateCollection>();

  
  // now make the valuemaps

  fillValueMap<reco::GsfElectronCollection>(iEvent, 
					    nameOutputElectronsPF_, 
					    gsfElectrons, 
					    electronCandidateMap,
					    pfCandidates,
					    pfCandidateRefProd);
  
  fillValueMap<reco::PhotonCollection>(iEvent, 
				       nameOutputPhotonsPF_, 
				       photons, 
				       photonCandidateMap,
				       pfCandidates,
				       pfCandidateRefProd);

  if(fillMuonRefs_){
    edm::Handle<reco::MuonCollection> muons; 
    iEvent.getByLabel(inputTagMuons_.label(), muons);
    
    fillValueMap<reco::MuonCollection>(iEvent, 
				       inputTagMuons_.label(), 
				       muons, 
				       muonCandidateMap,
				       pfCandidates,
				       pfCandidateRefProd);
  }
}

template<typename T>
bool PFLinker::fetchCollection(edm::Handle<T>& c, 
			       const edm::InputTag& tag, 
			       const edm::Event& iEvent) const {  

  bool found = iEvent.getByLabel(tag, c);
  
  if(!found )
    {
      std::ostringstream  err;
      err<<" cannot get " <<tag<<std::endl;
      edm::LogError("PFLinker")<<err.str();
    }
  return found;
}



template<typename TYPE>
void PFLinker::fillValueMap(edm::Event & event,
			    std::string label,
			    edm::Handle<TYPE>& inputObjCollection,
			    const std::map<edm::Ref<TYPE>, unsigned> & mapToTheCandidate,
			    const edm::Handle<reco::PFCandidateCollection> & oldPFCandColl,
			    const edm::OrphanHandle<reco::PFCandidateCollection> & newPFCandColl) const {

  std::auto_ptr<edm::ValueMap<reco::PFCandidatePtr> > pfMap_p(new edm::ValueMap<reco::PFCandidatePtr>());
  edm::ValueMap<reco::PFCandidatePtr>::Filler filler(*pfMap_p);

  typedef typename std::map<edm::Ref<TYPE>, unsigned int>::const_iterator MapTYPE_it; 

  unsigned nObj=inputObjCollection->size();
  std::vector<reco::PFCandidatePtr> values(nObj);

  for(unsigned iobj=0; iobj < nObj; ++iobj) {

    edm::Ref<TYPE> objRef(inputObjCollection, iobj);
    MapTYPE_it  itcheck = mapToTheCandidate.find(objRef);

    reco::PFCandidatePtr candPtr;

    if(itcheck != mapToTheCandidate.end()){
      if(producePFCandidates_)
	// points to the new collection
	candPtr = reco::PFCandidatePtr(newPFCandColl,itcheck->second);
      else 
	// points to the old collection
	candPtr = reco::PFCandidatePtr(oldPFCandColl,itcheck->second);
    }
    values[iobj] = candPtr;    
  }

  filler.insert(inputObjCollection,values.begin(),values.end());
  filler.fill();
  event.put(pfMap_p,label);
}
