//
// $Id: PATElectronProducer.cc,v 1.5 2008/01/22 13:23:09 lowette Exp $
//

#include "PhysicsTools/PatAlgos/interface/PATElectronProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "PhysicsTools/Utilities/interface/DeltaR.h"

#include "PhysicsTools/PatUtils/interface/LeptonLRCalc.h"
#include "PhysicsTools/PatUtils/interface/ObjectResolutionCalc.h"
#include "PhysicsTools/PatUtils/interface/TrackerIsolationPt.h"
#include "PhysicsTools/PatUtils/interface/CaloIsolationEnergy.h"

#include <vector>
#include <memory>


using namespace pat;


PATElectronProducer::PATElectronProducer(const edm::ParameterSet & iConfig) {

  // general configurables
  electronSrc_      = iConfig.getParameter<edm::InputTag>( "electronSource" );
  // ghost removal configurable
  doGhostRemoval_   = iConfig.getParameter<bool>         ( "removeDuplicates" );
  // MC matching configurables
  addGenMatch_      = iConfig.getParameter<bool>         ( "addGenMatch" );
  genPartSrc_       = iConfig.getParameter<edm::InputTag>( "genParticleSource" );
  maxDeltaR_        = iConfig.getParameter<double>       ( "maxDeltaR" );
  minRecoOnGenEt_   = iConfig.getParameter<double>       ( "minRecoOnGenEt" );
  maxRecoOnGenEt_   = iConfig.getParameter<double>       ( "maxRecoOnGenEt" );
  // resolution configurables
  addResolutions_   = iConfig.getParameter<bool>         ( "addResolutions" );
  useNNReso_        = iConfig.getParameter<bool>         ( "useNNResolutions" );
  electronResoFile_ = iConfig.getParameter<std::string>  ( "electronResoFile" );
  // isolation configurables
  addTrkIso_        = iConfig.getParameter<bool>         ( "addTrkIsolation" );
  tracksSrc_        = iConfig.getParameter<edm::InputTag>( "tracksSource" );
  addCalIso_        = iConfig.getParameter<bool>         ( "addCalIsolation" );
  towerSrc_         = iConfig.getParameter<edm::InputTag>( "towerSource" );
  // electron ID configurables
  addElecID_        = iConfig.getParameter<bool>         ( "addElectronID" );
  elecIDSrc_        = iConfig.getParameter<edm::InputTag>( "electronIDSource" );
  addElecIDRobust_  = iConfig.getParameter<bool>         ( "addElectronIDRobust" );
  elecIDRobustSrc_  = iConfig.getParameter<edm::InputTag>( "electronIDRobustSource" );
  
  // likelihood ratio configurables
  addLRValues_      = iConfig.getParameter<bool>         ( "addLRValues" );
  electronLRFile_   = iConfig.getParameter<std::string>  ( "electronLRFile" );
  // configurables for isolation from egamma producer
  addEgammaIso_     = iConfig.getParameter<bool>         ( "addEgammaIso");
  egammaTkIsoSrc_   = iConfig.getParameter<edm::InputTag>( "egammaTkIsoSource");
  egammaTkNumIsoSrc_= iConfig.getParameter<edm::InputTag>( "egammaTkNumIsoSource");
  egammaEcalIsoSrc_ = iConfig.getParameter<edm::InputTag>( "egammaEcalIsoSource");
  egammaHcalIsoSrc_ = iConfig.getParameter<edm::InputTag>( "egammaHcalIsoSource");
  
  // construct resolution calculator
  if(addResolutions_){
    theResoCalc_= new ObjectResolutionCalc(edm::FileInPath(electronResoFile_).fullPath(), useNNReso_);
  }

  // produces vector of muons
  produces<std::vector<Electron> >();

}


PATElectronProducer::~PATElectronProducer() {
  if(addResolutions_) delete theResoCalc_;
}


void PATElectronProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {

  // Get the collection of electrons from the event
  edm::Handle<edm::View<ElectronType> > electrons;
  iEvent.getByLabel(electronSrc_, electrons);


  // prepare the MC matching
  edm::Handle<edm::View<reco::Candidate> > particles;
  if (addGenMatch_) {
    iEvent.getByLabel(genPartSrc_, particles);
    matchTruth(*particles, *electrons) ;
  }

  // prepare isolation calculation
  edm::Handle<edm::View<reco::Track> > tracks;
  if (addTrkIso_) {
    trkIsolation_= new TrackerIsolationPt();
    iEvent.getByLabel(tracksSrc_, tracks);
  }
  edm::Handle<reco::CandViewDoubleAssociations> tkIso;
  edm::Handle<reco::CandViewDoubleAssociations> tkNumIso;
  edm::Handle<reco::CandViewDoubleAssociations> ecalIso;
  edm::Handle<reco::CandViewDoubleAssociations> hcalIso;
  if (addEgammaIso_) {
    iEvent.getByLabel(egammaTkIsoSrc_,tkIso);
    iEvent.getByLabel(egammaTkNumIsoSrc_,tkNumIso);
    iEvent.getByLabel(egammaEcalIsoSrc_,ecalIso);
    iEvent.getByLabel(egammaHcalIsoSrc_,hcalIso);
  }
  std::vector<CaloTower> towers;
  if (addCalIso_) {
    calIsolation_= new CaloIsolationEnergy();
    edm::Handle<edm::View<CaloTower> > towersH;
    iEvent.getByLabel(towerSrc_, towersH);
    for (edm::View<CaloTower>::const_iterator itTower = towersH->begin(); itTower != towersH->end(); itTower++) {
      towers.push_back(*itTower);
    }
  }
  
  // prepare ID extraction
  edm::Handle<reco::ElectronIDAssociationCollection> elecIDs;
  if (addElecID_) iEvent.getByLabel(elecIDSrc_, elecIDs);
  edm::Handle<reco::ElectronIDAssociationCollection> elecIDRobusts;
  if (addElecIDRobust_) iEvent.getByLabel(elecIDRobustSrc_, elecIDRobusts);
  
  // prepare LR calculation
  if(addLRValues_) {
    theLeptonLRCalc_= new LeptonLRCalc(iSetup, edm::FileInPath(electronLRFile_).fullPath(), "", "");
  }

  std::vector<Electron> * patElectrons = new std::vector<Electron>();
  for (edm::View<ElectronType>::const_iterator itElectron = electrons->begin(); itElectron != electrons->end(); ++itElectron) {
    // construct the Electron from the ref -> save ref to original object
    unsigned int idx = itElectron - electrons->begin();
    edm::Ref<std::vector<ElectronType> > elecsRef = electrons->refAt(idx).castTo<edm::Ref<std::vector<ElectronType> > >();
    Electron anElectron(elecsRef);
    // match to generated final state electrons
    if (addGenMatch_) {
      anElectron.setGenLepton(findTruth(*particles, *itElectron));
    }
    // add resolution info
    if(addResolutions_){
      (*theResoCalc_)(anElectron);
    }
    // do tracker isolation
    if (addTrkIso_) {
      anElectron.setTrackIso(trkIsolation_->calculate(anElectron, *tracks));
    }
    // do calorimeter isolation
    if (addCalIso_) {
      anElectron.setCaloIso(calIsolation_->calculate(anElectron, towers));
    }
    // add isolation from egamma producers
    if (addEgammaIso_) {
      setEgammaIso(anElectron, electrons, tkIso, tkNumIso, ecalIso, hcalIso, idx);
    }
    // add electron ID info
    if (addElecID_) {
      anElectron.setLeptonID(electronID(electrons, elecIDs, idx));
    }
    if (addElecIDRobust_) {
      anElectron.setElectronIDRobust(electronID(electrons, elecIDRobusts, idx));
    }
    // add lepton LR info
    if (addLRValues_) {
      theLeptonLRCalc_->calcLikelihood(anElectron, tracks, iEvent);
    }
    // add sel to selected
    patElectrons->push_back(anElectron);
  }

  
  // remove ghosts
  if (doGhostRemoval_) {
    removeEleDupes(patElectrons);
    //removeGhosts(electrons);has bug, replaced with clunkier but working code.
  }


  // sort electrons in pt
  std::sort(patElectrons->begin(), patElectrons->end(), pTComparator_);

  // add the electrons to the event output
  std::auto_ptr<std::vector<Electron> > ptr(patElectrons);
  iEvent.put(ptr);

  // clean up
  if (addTrkIso_) delete trkIsolation_;
  if (addCalIso_) delete calIsolation_;
  if (addLRValues_) delete theLeptonLRCalc_;

}


void PATElectronProducer::removeGhosts(std::vector<ElectronType> & elecs) {
  std::vector<ElectronType>::iterator cmp = elecs.begin();  
  std::vector<ElectronType>::iterator ref = elecs.begin();  


  for( ; ref<elecs.end(); ++ref ){
    for( ; (cmp!=ref) && cmp<elecs.end(); ++cmp ){
      
      if ((cmp->gsfTrack()==ref->gsfTrack()) || (cmp->superCluster()==ref->superCluster()) ){
	//same track or super cluster is used
	//keep the one with E/p closer to one	
	if(fabs(ref->eSuperClusterOverP()-1.) < fabs(cmp->eSuperClusterOverP()-1.)){
	  elecs.erase( cmp );
	} 
	else{
	  elecs.erase( ref );
	}
      }
    }
  }
  return;
}


reco::GenParticleCandidate PATElectronProducer::findTruth(const edm::View<reco::Candidate> & parts, const ElectronType & elec) {
  reco::GenParticleCandidate theGenElectron(0, reco::Particle::LorentzVector(0,0,0,0), reco::Particle::Point(0,0,0), 0, 0, true);
  for(std::vector<std::pair<const reco::Candidate *, const ElectronType *> >::const_iterator pairGenRecoElectrons = pairGenRecoElectronsVector_.begin(); pairGenRecoElectrons != pairGenRecoElectronsVector_.end(); ++pairGenRecoElectrons){
    if (fabs(elec.pt() - (pairGenRecoElectrons->second)->pt()) < 0.00001) {
      theGenElectron = *(dynamic_cast<reco::GenParticleCandidate *>(const_cast<reco::Candidate *>(pairGenRecoElectrons->first)));
    }
  }
  return theGenElectron;
}


void PATElectronProducer::matchTruth(const edm::View<reco::Candidate> & particles, const edm::View<ElectronType> & electrons) {
  pairGenRecoElectronsVector_.clear();
  for(edm::View<reco::Candidate>::const_iterator itGenElectron = particles.begin(); itGenElectron != particles.end(); ++itGenElectron) {
    reco::GenParticleCandidate aGenElectron = *(dynamic_cast<reco::GenParticleCandidate *>(const_cast<reco::Candidate *>(&*itGenElectron)));
    if (abs(aGenElectron.pdgId())==11 && aGenElectron.status()==1){
      const ElectronType * bestRecoElectron = 0;
      bool recoElectronFound = false;
      float bestDR = 100000;
      //loop over reconstructed electrons
      for (edm::View<ElectronType>::const_iterator itElectron = electrons.begin(); itElectron != electrons.end(); ++itElectron) {
	float recoEtOnGenEt = itElectron->et()/aGenElectron.et();
	// if the charge is the same and the energy comparable
	//FIXME set recoEtOnGenEt cut configurable 
	  float currDR = DeltaR<reco::Candidate>()(aGenElectron, *itElectron);
	  //if ( aGenElectron.charge()==itElectron->charge() && recoEtOnGenEt > minRecoOnGenEt_ 
	  //     &&  recoEtOnGenEt < maxRecoOnGenEt_ && currDR < maxDeltaR_ ) {
	  if (  recoEtOnGenEt > minRecoOnGenEt_ 
		&&  recoEtOnGenEt < maxRecoOnGenEt_ && currDR < maxDeltaR_ ) {
	    //if the reco electrons is the closest one
	    if ( currDR < bestDR) {
	      bestRecoElectron = &(*itElectron);
	      bestDR = currDR;
	      recoElectronFound = true;
	    }
	  }
      }
      if(recoElectronFound == true){
	pairGenRecoElectronsVector_.push_back(std::pair<const reco::Candidate *, const ElectronType *>(&*itGenElectron, bestRecoElectron));
      }
    }
  }
}


double PATElectronProducer::electronID(const edm::Handle<edm::View<ElectronType> > & electrons,
                                       const edm::Handle<reco::ElectronIDAssociationCollection> & elecIDs,
	                               unsigned int idx) {
  //find elecID for elec with index idx
  edm::Ref<std::vector<ElectronType> > elecsRef = electrons->refAt(idx).castTo<edm::Ref<std::vector<ElectronType> > >();
  reco::ElectronIDAssociationCollection::const_iterator elecID = elecIDs->find( elecsRef );

  //return corresponding elecID (only 
  //cut based available at the moment)
  const reco::ElectronIDRef& id = elecID->val;
  return id->cutBasedDecision();
}


//fill the Electron with the isolation quantities calculated by the egamma producers
void PATElectronProducer::setEgammaIso(Electron & anElectron,
                                    const edm::Handle<edm::View<ElectronType> > & electrons,
                                    const edm::Handle<reco::CandViewDoubleAssociations> tkIso,
                                    const edm::Handle<reco::CandViewDoubleAssociations> tkNumIso,
                                    const edm::Handle<reco::CandViewDoubleAssociations> ecalIso,
                                    const edm::Handle<reco::CandViewDoubleAssociations> hcalIso,
                                    unsigned int idx) {
  //find isolations for elec with index idx
  edm::Ref<std::vector<ElectronType> > elecsRef = electrons->refAt(idx).castTo<edm::Ref<std::vector<ElectronType> > >();
  reco::CandidateBaseRef candRef(elecsRef);
  anElectron.setEgammaTkIso((*tkIso)[candRef]);
  anElectron.setEgammaTkNumIso((int) (*tkNumIso)[candRef]);
  anElectron.setEgammaEcalIso((*ecalIso)[candRef]);
  anElectron.setEgammaHcalIso((*hcalIso)[candRef]);
}


//it is possible that there are multiple electron objects in the collection that correspond to the same
//real physics object - a supercluster with two tracks reconstructed to it, or a track that points to two different SC
// (i would guess the latter doesn't actually happen).
//NB triplicates also appear in the electron collection provided by egamma group, it is necessary to handle those correctly

//this function removes the duplicates/triplicates/multiplicates from the input vector
void PATElectronProducer::removeEleDupes(std::vector<Electron> * electrons) {
  
  //contains indices of duplicate electrons marked for removal
  //I do it this way because removal during the loop is more confusing
  std::vector<size_t> indicesToRemove;
  
  for (size_t ie=0;ie<electrons->size();ie++) {
    if (find(indicesToRemove.begin(),indicesToRemove.end(),ie)!=indicesToRemove.end()) continue;//ignore if already marked for removal
    
    reco::GsfTrackRef thistrack=electrons->at(ie).gsfTrack();
    reco::SuperClusterRef thissc=electrons->at(ie).superCluster();

    for (size_t je=ie+1;je<electrons->size();je++) {
      if (find(indicesToRemove.begin(),indicesToRemove.end(),je)!=indicesToRemove.end()) continue;//ignore if already marked for removal
      
      if ((thistrack==electrons->at(je).gsfTrack()) ||
	  (thissc==electrons->at(je).superCluster()) ) {//we have a match, arbitrate and mark one for removal
	//keep the one with E/P closer to unity
	float diff1=fabs(electrons->at(ie).eSuperClusterOverP()-1);
	float diff2=fabs(electrons->at(je).eSuperClusterOverP()-1);
	
	if (diff1<diff2) {
	  indicesToRemove.push_back(je);
	} else {
	  indicesToRemove.push_back(ie);
	}
      }
    }
  }
  //now remove the ones marked
  //or in fact, copy the old vector into a tmp vector, skipping the ones that are duplicates,
  //then clear the original and copy back the contents of the tmp
  //this is ugly but it will work
  std::vector<Electron> tmp;
  for (size_t ie=0;ie<electrons->size();ie++) {
    if (find(indicesToRemove.begin(),indicesToRemove.end(),ie)!=indicesToRemove.end()) {
      continue;
    } else {
      tmp.push_back(electrons->at(ie));
    }
  }
  //copy back
  electrons->clear();
  electrons->assign(tmp.begin(),tmp.end());
  
  return;
}

