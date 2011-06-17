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
  // vector of InputTag; this is not for RECO, it is for analysis
  inputTagPFCandidates_ 
    = iConfig.getParameter<std::vector<edm::InputTag> >("PFCandidates");

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

  nameOutputMergedPF_ 
    = iConfig.getParameter<std::string>("ValueMapMerged");

  producePFCandidates_  
    = iConfig.getParameter<bool>("ProducePFCandidates");
  
  fillMuonRefs_
    = iConfig.getParameter<bool>("FillMuonRefs");

  // should not produce PFCandidates and read seve
  if(producePFCandidates_ && inputTagPFCandidates_.size()>1) {
    edm::LogError("PFLinker") << " cannot read several collections of PFCandidates and produce a new collection at the same time. " << std::endl;
    assert(0);
  }

  if(producePFCandidates_) {
    produces<reco::PFCandidateCollection>(nameOutputPF_);
  }
  produces<edm::ValueMap<reco::PFCandidatePtr> > (nameOutputElectronsPF_);
  produces<edm::ValueMap<reco::PFCandidatePtr> > (nameOutputPhotonsPF_);
  if(fillMuonRefs_)  produces<edm::ValueMap<reco::PFCandidatePtr> > (inputTagMuons_.label());
  produces<edm::ValueMap<reco::PFCandidatePtr> > (nameOutputMergedPF_);
}

PFLinker::~PFLinker() {;}

void PFLinker::beginRun(edm::Run& run,const edm::EventSetup & es) {;}

void PFLinker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
  std::auto_ptr<reco::PFCandidateCollection>
    pfCandidates_p(new reco::PFCandidateCollection);
  
  std::auto_ptr<edm::ValueMap<reco::PFCandidatePtr> > 
    pfMapGsfElectrons_p(new edm::ValueMap<reco::PFCandidatePtr>());
  edm::ValueMap<reco::PFCandidatePtr>::Filler pfMapGsfElectronFiller(*pfMapGsfElectrons_p);
  
  std::auto_ptr<edm::ValueMap<reco::PFCandidatePtr> > 
    pfMapPhotons_p(new edm::ValueMap<reco::PFCandidatePtr>());
  edm::ValueMap<reco::PFCandidatePtr>::Filler pfMapPhotonFiller(*pfMapPhotons_p);

  std::auto_ptr<edm::ValueMap<reco::PFCandidatePtr> > 
    pfMapMuons_p(new edm::ValueMap<reco::PFCandidatePtr>());
  edm::ValueMap<reco::PFCandidatePtr>::Filler pfMapMuonFiller(*pfMapMuons_p);

  std::auto_ptr<edm::ValueMap<reco::PFCandidatePtr> > 
    pfMapMerged_p(new edm::ValueMap<reco::PFCandidatePtr>());
  edm::ValueMap<reco::PFCandidatePtr>::Filler pfMapMergedFiller(*pfMapMerged_p);

  edm::Handle<reco::GsfElectronCollection> gsfElectrons;
  bool status=fetchCollection<reco::GsfElectronCollection>(gsfElectrons,
						      inputTagGsfElectrons_,
						      iEvent );
  std::map<reco::GsfElectronRef,reco::PFCandidatePtr> electronCandidateMap;  

  edm::Handle<reco::PhotonCollection> photons;
  status=fetchCollection<reco::PhotonCollection>(photons,
						 inputTagPhotons_,
						 iEvent );
  std::map<reco::PhotonRef,reco::PFCandidatePtr> photonCandidateMap;


  edm::Handle<reco::MuonToMuonMap> muonMap;
  if(fillMuonRefs_)
    status=fetchCollection<reco::MuonToMuonMap>(muonMap,
						inputTagMuons_,
						iEvent);
  std::map<reco::MuonRef,reco::PFCandidatePtr> muonCandidateMap;

  unsigned nColPF=inputTagPFCandidates_.size();
  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  for(unsigned icol=0;icol<nColPF;++icol) {

    bool status=fetchCollection<reco::PFCandidateCollection>(pfCandidates, 
							     inputTagPFCandidates_[icol], 
							     iEvent );
    
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
	muonCandidateMap[muRef] = candPtr;
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
	electronCandidateMap[electronRef]=candPtr;
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
	photonCandidateMap[photonRef]=candPtr;
      }      
      pfCandidates_p->push_back(cand);      
    }
    // save the PFCandidates and get a valid handle

  }
  const edm::OrphanHandle<reco::PFCandidateCollection> pfCandidateRefProd = (producePFCandidates_) ? iEvent.put(pfCandidates_p,nameOutputPF_) :
    edm::OrphanHandle<reco::PFCandidateCollection>();
  
  // now make the valuemaps  
  fillValueMap<reco::GsfElectronCollection>(gsfElectrons, 
					    electronCandidateMap,
					    pfCandidateRefProd,
					    pfMapGsfElectronFiller);
  
  fillValueMap<reco::PhotonCollection>(photons, 
				       photonCandidateMap,
				       pfCandidateRefProd,
				       pfMapPhotonFiller);
  
  if(fillMuonRefs_){
    edm::Handle<reco::MuonCollection> muons; 
    iEvent.getByLabel(inputTagMuons_.label(), muons);
    
    fillValueMap<reco::MuonCollection>(muons, 
				       muonCandidateMap,
				       pfCandidateRefProd,
				       pfMapMuonFiller);
  }
  
  // Finalize the ValueMaps
  pfMapGsfElectronFiller.fill();
  pfMapPhotonFiller.fill();  
  if(fillMuonRefs_) {
    pfMapMuonFiller.fill();
  }
  
  // The merged value map
  *pfMapMerged_p+=*pfMapGsfElectrons_p;
  *pfMapMerged_p+=(*pfMapPhotons_p);
  if(fillMuonRefs_)
    *pfMapMerged_p+=(*pfMapMuons_p);

  // now save in the event
  iEvent.put(pfMapGsfElectrons_p,nameOutputElectronsPF_);
  iEvent.put(pfMapPhotons_p,nameOutputPhotonsPF_);
  iEvent.put(pfMapMerged_p,nameOutputMergedPF_);
  if(fillMuonRefs_) {
    iEvent.put(pfMapMuons_p,inputTagMuons_.label());
  }

  // clean up
  electronCandidateMap.clear();
  photonCandidateMap.clear();
  muonCandidateMap.clear();
  
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


template<typename TYPE>
void PFLinker::fillValueMap(edm::Handle<TYPE>& inputObjCollection,
			    const std::map<edm::Ref<TYPE>, reco::PFCandidatePtr> & mapToTheCandidate,
			    const edm::OrphanHandle<reco::PFCandidateCollection> & newPFCandColl,
			    edm::ValueMap<reco::PFCandidatePtr>::Filler & filler ) const {


  typedef typename std::map<edm::Ref<TYPE>, reco::PFCandidatePtr>::const_iterator MapTYPE_it; 

  unsigned nObj=inputObjCollection->size();
  std::vector<reco::PFCandidatePtr> values(nObj);

  for(unsigned iobj=0; iobj < nObj; ++iobj) {

    edm::Ref<TYPE> objRef(inputObjCollection, iobj);
    MapTYPE_it  itcheck = mapToTheCandidate.find(objRef);

    reco::PFCandidatePtr candPtr;

    if(itcheck != mapToTheCandidate.end()){
      if(producePFCandidates_)
	// points to the new collection
	candPtr = reco::PFCandidatePtr(newPFCandColl,itcheck->second.key());
      else 
	// points to the old collection
	candPtr = itcheck->second;
    }
    values[iobj] = candPtr;    
  }
  filler.insert(inputObjCollection,values.begin(),values.end());
}
