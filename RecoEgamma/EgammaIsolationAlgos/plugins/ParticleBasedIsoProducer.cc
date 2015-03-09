#include "RecoEgamma/EgammaIsolationAlgos/plugins/ParticleBasedIsoProducer.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/ValueMap.h"



ParticleBasedIsoProducer::ParticleBasedIsoProducer(const edm::ParameterSet& conf) : conf_(conf) {

  photonTmpProducer_       = conf_.getParameter<edm::InputTag>("photonTmpProducer");
  photonProducer_       = conf_.getParameter<edm::InputTag>("photonProducer");
  electronProducer_     = conf_.getParameter<edm::InputTag>("electronProducer");
  electronTmpProducer_     = conf_.getParameter<edm::InputTag>("electronTmpProducer");

  photonProducerT_   = 
    consumes<reco::PhotonCollection>(photonProducer_);


  photonTmpProducerT_   = 
    consumes<reco::PhotonCollection>(photonTmpProducer_);


  electronProducerT_   = 
    consumes<reco::GsfElectronCollection>(electronProducer_);

  electronTmpProducerT_   = 
    consumes<reco::GsfElectronCollection>(electronTmpProducer_);

  pfCandidates_      = 
    consumes<reco::PFCandidateCollection>(conf_.getParameter<edm::InputTag>("pfCandidates"));
  
  pfEgammaCandidates_      = 
    consumes<reco::PFCandidateCollection>(conf_.getParameter<edm::InputTag>("pfEgammaCandidates"));

  valueMapPFCandPhoton_ = conf_.getParameter<std::string>("valueMapPhoToEG");  
  valueMapPFCandEle_    = conf_.getParameter<std::string>("valueMapEleToEG");

  valMapPFCandToPhoton_ = 
    consumes<edm::ValueMap<reco::PhotonRef> >(edm::InputTag("gedPhotonsTmp",valueMapPFCandPhoton_));

  valMapPFCandToEle_ = 
    consumes<edm::ValueMap<reco::GsfElectronRef> >(edm::InputTag("gedGsfElectronsTmp",valueMapPFCandEle_));

  valueMapPhoPFCandIso_ = conf_.getParameter<std::string>("valueMapPhoPFblockIso");
  valueMapElePFCandIso_ = conf_.getParameter<std::string>("valueMapElePFblockIso");


  produces< edm::ValueMap<std::vector<reco::PFCandidateRef> > >  (valueMapPhoPFCandIso_); 
  produces< edm::ValueMap<std::vector<reco::PFCandidateRef> > > (valueMapElePFCandIso_); 

}

ParticleBasedIsoProducer::~ParticleBasedIsoProducer() {


}


void ParticleBasedIsoProducer::beginRun(const edm::Run & run, const edm::EventSetup& c) {

    thePFBlockBasedIsolation_ = new PFBlockBasedIsolation();
    edm::ParameterSet pfBlockBasedIsolationSetUp = conf_.getParameter<edm::ParameterSet>("pfBlockBasedIsolationSetUp"); 
    thePFBlockBasedIsolation_ ->setup(pfBlockBasedIsolationSetUp);

}

void ParticleBasedIsoProducer::endRun(const edm::Run & run, const edm::EventSetup& c) {

  delete thePFBlockBasedIsolation_;

}



void ParticleBasedIsoProducer::produce(edm::Event& theEvent, const edm::EventSetup& c) {

  edm::Handle<reco::PhotonCollection> photonHandle;
  theEvent.getByToken(photonProducerT_,photonHandle);

  edm::Handle<reco::PhotonCollection> photonTmpHandle;
  theEvent.getByToken(photonTmpProducerT_,photonTmpHandle);

  edm::Handle<reco::GsfElectronCollection> electronTmpHandle;
  theEvent.getByToken(electronTmpProducerT_,electronTmpHandle);

  edm::Handle<reco::GsfElectronCollection> electronHandle;
  theEvent.getByToken(electronProducerT_,electronHandle);
  
  edm::Handle<reco::PFCandidateCollection> pfEGCandidateHandle;
  // Get the  PF refined cluster  collection
  theEvent.getByToken(pfEgammaCandidates_,pfEGCandidateHandle);
  
  edm::Handle<reco::PFCandidateCollection> pfCandidateHandle;
  // Get the  PF candidates collection
  theEvent.getByToken(pfCandidates_,pfCandidateHandle);
  
  edm::ValueMap<reco::PhotonRef> pfEGCandToPhotonMap;
  edm::Handle<edm::ValueMap<reco::PhotonRef> > pfEGCandToPhotonMapHandle;
  theEvent.getByToken(valMapPFCandToPhoton_,pfEGCandToPhotonMapHandle);
  pfEGCandToPhotonMap = *(pfEGCandToPhotonMapHandle.product());

  edm::ValueMap<reco::GsfElectronRef> pfEGCandToElectronMap;
  edm::Handle<edm::ValueMap<reco::GsfElectronRef> > pfEGCandToElectronMapHandle;
  theEvent.getByToken(valMapPFCandToEle_,pfEGCandToElectronMapHandle);
  pfEGCandToElectronMap = *(pfEGCandToElectronMapHandle.product());

  std::vector<std::vector<reco::PFCandidateRef>> pfCandIsoPairVecPho;

  ///// Isolation for photons 
  //  std::cout << " ParticleBasedIsoProducer  photonHandle size " << photonHandle->size() << std::endl;
  for(unsigned int lSC=0; lSC < photonTmpHandle->size(); lSC++) {

    reco::PhotonRef phoRef(reco::PhotonRef(photonTmpHandle, lSC));

    // loop over the unbiased candidates to retrieve the ref to the unbiased candidate corresponding to this photon
    unsigned nObj = pfEGCandidateHandle->size();
    reco::PFCandidateRef pfEGCandRef;

    std::vector<reco::PFCandidateRef> pfCandIsoPairPho;
    for(unsigned int lCand=0; lCand < nObj; lCand++) {
      pfEGCandRef=reco::PFCandidateRef(pfEGCandidateHandle,lCand);
      reco::PhotonRef myPho= (pfEGCandToPhotonMap)[pfEGCandRef];
      
      if ( myPho.isNonnull() ) {
	//std::cout << "ParticleBasedIsoProducer photons PF SC " << pfEGCandRef->superClusterRef()->energy() << " Photon SC " << myPho->superCluster()->energy() << std::endl;
	if (myPho != phoRef) continue;
	//	std::cout << " ParticleBasedIsoProducer photons This is my egammaunbiased guy energy " <<  pfEGCandRef->superClusterRef()->energy() << std::endl;
	pfCandIsoPairPho=thePFBlockBasedIsolation_->calculate (myPho->p4(),  pfEGCandRef, pfCandidateHandle);
	
	/////// debug
	//	for ( std::vector<reco::PFCandidateRef>::const_iterator iPair=pfCandIsoPairPho.begin(); iPair<pfCandIsoPairPho.end(); iPair++) {
	// float dR= deltaR(myPho->eta(),  myPho->phi(), (*iPair)->eta(),  (*iPair)->phi() );
	// std::cout << " ParticleBasedIsoProducer photons  checking the pfCand bool pair " << (*iPair)->particleId() << " dR " << dR << " pt " <<  (*iPair)->pt() << std::endl; 
	//	}
	
	
      }
      
    }
    
    pfCandIsoPairVecPho.push_back(pfCandIsoPairPho); 
  }
 


  ////////////isolation for electrons 
  std::vector<std::vector<reco::PFCandidateRef>> pfCandIsoPairVecEle;
  //  std::cout << " ParticleBasedIsoProducer  electronHandle size " << electronHandle->size() << std::endl;
  for(unsigned int lSC=0; lSC < electronTmpHandle->size(); lSC++) {
    reco::GsfElectronRef eleRef(reco::GsfElectronRef(electronTmpHandle, lSC));
    
    // loop over the unbiased candidates to retrieve the ref to the unbiased candidate corresponding to this electron
    unsigned nObj = pfEGCandidateHandle->size();
    reco::PFCandidateRef pfEGCandRef;
    
    std::vector<reco::PFCandidateRef> pfCandIsoPairEle;
    for(unsigned int lCand=0; lCand < nObj; lCand++) {
      pfEGCandRef=reco::PFCandidateRef(pfEGCandidateHandle,lCand);
      reco::GsfElectronRef myEle= (pfEGCandToElectronMap)[pfEGCandRef];
      
      if ( myEle.isNonnull() ) {
	//	std::cout << "ParticleBasedIsoProducer Electorns PF SC " << pfEGCandRef->superClusterRef()->energy() << " Electron SC " << myEle->superCluster()->energy() << std::endl;
	if (myEle != eleRef) continue;
	
	//math::XYZVector candidateMomentum(myEle->p4().px(),myEle->p4().py(),myEle->p4().pz());
	//math::XYZVector myDir=candidateMomentum.Unit();
	//	std::cout << " ParticleBasedIsoProducer  Electrons This is my egammaunbiased guy energy " <<  pfEGCandRef->superClusterRef()->energy()  << std::endl;
	//  std::cout << " Ele  direction " << myDir << " eta " << myEle->eta() << " phi " << myEle->phi() << std::endl;
	pfCandIsoPairEle=thePFBlockBasedIsolation_->calculate (myEle->p4(),  pfEGCandRef, pfCandidateHandle);
	/////// debug
	//for ( std::vector<reco::PFCandidateRef>::const_iterator iPair=pfCandIsoPairEle.begin(); iPair<pfCandIsoPairEle.end(); iPair++) {
	// float dR= deltaR(myEle->eta(),  myEle->phi(), (*iPair)->eta(),  (*iPair)->phi() );
	// std::cout << " ParticleBasedIsoProducer Electron  checking the pfCand bool pair " << (*iPair)->particleId() << " dR " << dR << " pt " <<  (*iPair)->pt() << " eta " << (*iPair)->eta() << " phi " << (*iPair)->phi() <<  std::endl; 
	//	}
	
	
      }
      
    }
    
    pfCandIsoPairVecEle.push_back(pfCandIsoPairEle); 
}
 




  std::auto_ptr<edm::ValueMap<std::vector<reco::PFCandidateRef>> >  
    phoToPFCandIsoMap_p(new edm::ValueMap<std::vector<reco::PFCandidateRef>>());
  edm::ValueMap<std::vector<reco::PFCandidateRef>>::Filler 
    fillerPhotons(*phoToPFCandIsoMap_p);
  
  //// fill the isolation value map for photons
  fillerPhotons.insert(photonHandle,pfCandIsoPairVecPho.begin(),pfCandIsoPairVecPho.end());
  fillerPhotons.fill(); 
  theEvent.put(phoToPFCandIsoMap_p,valueMapPhoPFCandIso_);


  std::auto_ptr<edm::ValueMap<std::vector<reco::PFCandidateRef>> >  
    eleToPFCandIsoMap_p(new edm::ValueMap<std::vector<reco::PFCandidateRef>>());
  edm::ValueMap<std::vector<reco::PFCandidateRef>>::Filler 
    fillerElectrons(*eleToPFCandIsoMap_p);
  
  //// fill the isolation value map for electrons 
  fillerElectrons.insert(electronHandle,pfCandIsoPairVecEle.begin(),pfCandIsoPairVecEle.end());
  fillerElectrons.fill(); 
  theEvent.put(eleToPFCandIsoMap_p,valueMapElePFCandIso_);









 
}
