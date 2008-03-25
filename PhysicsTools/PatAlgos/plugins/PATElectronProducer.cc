//
// $Id: PATElectronProducer.cc,v 1.1 2008/03/06 09:23:10 llista Exp $
//

#include "PhysicsTools/PatAlgos/plugins/PATElectronProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "PhysicsTools/Utilities/interface/DeltaR.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

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
  // MC matching configurables
  addGenMatch_      = iConfig.getParameter<bool>          ( "addGenMatch" );
  genMatchSrc_       = iConfig.getParameter<edm::InputTag>( "genParticleMatch" );

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
  edm::Handle<edm::Association<reco::GenParticleCollection> > genMatch;
  if (addGenMatch_) {
    iEvent.getByLabel(genMatchSrc_, genMatch);
  }

  // prepare isolation calculation
  edm::Handle<edm::View<reco::Track> > tracks;
  if (addTrkIso_) {
    trkIsolation_= new TrackerIsolationPt();
    iEvent.getByLabel(tracksSrc_, tracks);
  }
  edm::Handle<edm::ValueMap<float> > tkIso;
  edm::Handle<edm::ValueMap<float> > tkNumIso;
  edm::Handle<edm::ValueMap<float> > ecalIso;
  edm::Handle<edm::ValueMap<float> > hcalIso;
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
    edm::RefToBase<ElectronType> elecsRef = electrons->refAt(idx);
    Electron anElectron(elecsRef);
    // match to generated final state electrons
    if (addGenMatch_) {
      reco::GenParticleRef genElectron = (*genMatch)[elecsRef];
      if (genElectron.isNonnull() && genElectron.isAvailable() ) {
        anElectron.setGenLepton(*genElectron);
      } else {
        // "MC ELE MATCH: Something wrong: null=" << !genElectron.isNonnull() <<", avail=" << genElectron.isAvailable() << std::endl;
        anElectron.setGenLepton(reco::Particle(0, reco::Particle::LorentzVector(0,0,0,0))); // TQAF way of setting "null"
      }
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
                                    const edm::Handle<edm::ValueMap<float> > tkIso,
                                    const edm::Handle<edm::ValueMap<float> > tkNumIso,
                                    const edm::Handle<edm::ValueMap<float> > ecalIso,
                                    const edm::Handle<edm::ValueMap<float> > hcalIso,
                                    unsigned int idx) {
  //find isolations for elec with index idx
  edm::Ref<std::vector<ElectronType> > elecsRef = electrons->refAt(idx).castTo<edm::Ref<std::vector<ElectronType> > >();
  reco::CandidateBaseRef candRef(elecsRef);
  anElectron.setEgammaTkIso((*tkIso)[candRef]);
  anElectron.setEgammaTkNumIso((int) (*tkNumIso)[candRef]);
  anElectron.setEgammaEcalIso((*ecalIso)[candRef]);
  anElectron.setEgammaHcalIso((*hcalIso)[candRef]);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATElectronProducer);
