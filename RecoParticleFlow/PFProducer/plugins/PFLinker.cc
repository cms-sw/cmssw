#include "RecoParticleFlow/PFProducer/plugins/PFLinker.h"


#include "RecoParticleFlow/PFProducer/interface/GsfElectronEqual.h"
#include "RecoParticleFlow/PFProducer/interface/PhotonEqual.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


PFLinker::PFLinker(const edm::ParameterSet & iConfig) {
  // vector of InputTag; more than 1 is not for RECO, it is for analysis
  
    std::vector<edm::InputTag> tags  = iConfig.getParameter<std::vector<edm::InputTag> >("PFCandidate");
  for (unsigned int i=0;i<tags.size();++i)
    inputTagPFCandidates_.push_back(consumes<reco::PFCandidateCollection>(tags[i]));   
  

  inputTagGsfElectrons_=consumes<reco::GsfElectronCollection>(
	      iConfig.getParameter<edm::InputTag>("GsfElectrons"));

  inputTagPhotons_=consumes<reco::PhotonCollection>(iConfig.getParameter<edm::InputTag>("Photons"));

  
  muonTag_ = iConfig.getParameter<edm::InputTag>("Muons");
  inputTagMuons_=consumes<reco::MuonCollection>(edm::InputTag(muonTag_.label()));
  inputTagMuonMap_=consumes<reco::MuonToMuonMap>(muonTag_);
  
  nameOutputPF_ 
    = iConfig.getParameter<std::string>("OutputPF");
  
  nameOutputElectronsPF_ 
    = iConfig.getParameter<std::string>("ValueMapElectrons");

  nameOutputPhotonsPF_ 
    = iConfig.getParameter<std::string>("ValueMapPhotons");

  producePFCandidates_  
    = iConfig.getParameter<bool>("ProducePFCandidates");

  nameOutputMergedPF_ 
    = iConfig.getParameter<std::string>("ValueMapMerged"); 
  
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
  produces<edm::ValueMap<reco::PFCandidatePtr> > (nameOutputMergedPF_);
  if(fillMuonRefs_)  produces<edm::ValueMap<reco::PFCandidatePtr> > (muonTag_.label());

}

PFLinker::~PFLinker() {;}

void PFLinker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
  std::auto_ptr<reco::PFCandidateCollection>
    pfCandidates_p(new reco::PFCandidateCollection);
  
  edm::Handle<reco::GsfElectronCollection> gsfElectrons;
  iEvent.getByToken(inputTagGsfElectrons_,gsfElectrons);

  std::map<reco::GsfElectronRef,reco::PFCandidatePtr> electronCandidateMap;  


  edm::Handle<reco::PhotonCollection> photons;
  iEvent.getByToken(inputTagPhotons_,photons);
  std::map<reco::PhotonRef,reco::PFCandidatePtr> photonCandidateMap;


  edm::Handle<reco::MuonToMuonMap> muonMap;
  if(fillMuonRefs_)
    iEvent.getByToken(inputTagMuonMap_,muonMap);
  std::map<reco::MuonRef,reco::PFCandidatePtr> muonCandidateMap;
  
  unsigned nColPF=inputTagPFCandidates_.size();
  
  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  for(unsigned icol=0;icol<nColPF;++icol) {
    iEvent.getByToken(inputTagPFCandidates_[icol],pfCandidates);
    unsigned ncand=pfCandidates->size();
    
    for( unsigned i=0; i<ncand; ++i) {
      edm::Ptr<reco::PFCandidate> candPtr(pfCandidates,i);
      reco::PFCandidate cand(candPtr);
      
      bool isphoton   = cand.particleId() == reco::PFCandidate::gamma && cand.mva_nothing_gamma()>0.;
      bool iselectron = cand.particleId() == reco::PFCandidate::e;
      // PFCandidates may have a valid MuonRef though they are not muons.
      bool hasNonNullMuonRef  = cand.muonRef().isNonnull() && fillMuonRefs_;
      
      // if not an electron or a photon or a muon just fill the PFCandidate collection
      if ( !(isphoton || iselectron || hasNonNullMuonRef)){pfCandidates_p->push_back(cand); continue;}
      
      
      if (hasNonNullMuonRef) {
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
	electronCandidateMap[electronRef] = candPtr;
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
	photonCandidateMap[photonRef] = candPtr;
      }      

      pfCandidates_p->push_back(cand);
    }      
    // save the PFCandidates and get a valid handle
  }
  const edm::OrphanHandle<reco::PFCandidateCollection> pfCandidateRefProd = (producePFCandidates_) ? iEvent.put(pfCandidates_p,nameOutputPF_) :
    edm::OrphanHandle<reco::PFCandidateCollection>();
  
  
  // now make the valuemaps

  edm::ValueMap<reco::PFCandidatePtr> pfMapGsfElectrons = fillValueMap<reco::GsfElectronCollection>(iEvent, 
								nameOutputElectronsPF_, 
								gsfElectrons, 
								electronCandidateMap,
								pfCandidateRefProd);
  
  edm::ValueMap<reco::PFCandidatePtr> pfMapPhotons = fillValueMap<reco::PhotonCollection>(iEvent, 
						      nameOutputPhotonsPF_, 
						      photons, 
						      photonCandidateMap,
						      pfCandidateRefProd);
  

  edm::ValueMap<reco::PFCandidatePtr> pfMapMuons;

  if(fillMuonRefs_){
    edm::Handle<reco::MuonCollection> muons; 
    iEvent.getByToken(inputTagMuons_, muons);
    
    pfMapMuons = fillValueMap<reco::MuonCollection>(iEvent, 
						    muonTag_.label(), 
						    muons, 
						    muonCandidateMap,
						    pfCandidateRefProd);
  }
  
  std::auto_ptr<edm::ValueMap<reco::PFCandidatePtr> > 
    pfMapMerged(new edm::ValueMap<reco::PFCandidatePtr>());
  edm::ValueMap<reco::PFCandidatePtr>::Filler pfMapMergedFiller(*pfMapMerged);
  
  *pfMapMerged                   += pfMapGsfElectrons;
  *pfMapMerged                   += pfMapPhotons;
  if(fillMuonRefs_) *pfMapMerged += pfMapMuons;
  
  iEvent.put(pfMapMerged,nameOutputMergedPF_);
}



template<typename TYPE>
edm::ValueMap<reco::PFCandidatePtr>  PFLinker::fillValueMap(edm::Event & event,
							    std::string label,
							    edm::Handle<TYPE>& inputObjCollection,
							    const std::map<edm::Ref<TYPE>, reco::PFCandidatePtr> & mapToTheCandidate,
							    const edm::OrphanHandle<reco::PFCandidateCollection> & newPFCandColl) const {
  
  std::auto_ptr<edm::ValueMap<reco::PFCandidatePtr> > pfMap_p(new edm::ValueMap<reco::PFCandidatePtr>());
  edm::ValueMap<reco::PFCandidatePtr>::Filler filler(*pfMap_p);
  
  typedef typename std::map<edm::Ref<TYPE>, reco::PFCandidatePtr>::const_iterator MapTYPE_it; 
  
  unsigned nObj=inputObjCollection->size();
  std::vector<reco::PFCandidatePtr> values(nObj);

  for(unsigned iobj=0; iobj < nObj; ++iobj) {

    edm::Ref<TYPE> objRef(inputObjCollection, iobj);
    MapTYPE_it  itcheck = mapToTheCandidate.find(objRef);

    reco::PFCandidatePtr candPtr;

    if(itcheck != mapToTheCandidate.end())
      candPtr = producePFCandidates_ ? reco::PFCandidatePtr(newPFCandColl,itcheck->second.key()) : itcheck->second;
    
    values[iobj] = candPtr;    
  }

  filler.insert(inputObjCollection,values.begin(),values.end());
  filler.fill();
  edm::ValueMap<reco::PFCandidatePtr> returnValue = *pfMap_p;
  event.put(pfMap_p,label);
  return returnValue;
}
