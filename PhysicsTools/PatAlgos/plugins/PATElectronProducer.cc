//
// $Id: PATElectronProducer.cc,v 1.2 2008/03/12 16:13:26 gpetrucc Exp $
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


PATElectronProducer::PATElectronProducer(const edm::ParameterSet & iConfig) :
  isolator_(iConfig.exists("isolation") ? iConfig.getParameter<edm::ParameterSet>("isolation") : edm::ParameterSet(), false) 
{

  // general configurables
  electronSrc_      = iConfig.getParameter<edm::InputTag>( "electronSource" );
  // MC matching configurables
  addGenMatch_      = iConfig.getParameter<bool>          ( "addGenMatch" );
  genMatchSrc_       = iConfig.getParameter<edm::InputTag>( "genParticleMatch" );

  // resolution configurables
  addResolutions_   = iConfig.getParameter<bool>         ( "addResolutions" );
  useNNReso_        = iConfig.getParameter<bool>         ( "useNNResolutions" );
  electronResoFile_ = iConfig.getParameter<std::string>  ( "electronResoFile" );

  // electron ID configurables
  addElecID_        = iConfig.getParameter<bool>         ( "addElectronID" );
  elecIDSrc_        = iConfig.getParameter<edm::InputTag>( "electronIDSource" );
  addElecIDRobust_  = iConfig.getParameter<bool>         ( "addElectronIDRobust" );
  elecIDRobustSrc_  = iConfig.getParameter<edm::InputTag>( "electronIDRobustSource" );
  
  // likelihood ratio configurables
  tracksSrc_        = iConfig.getParameter<edm::InputTag>( "tracksSource" );
  addLRValues_      = iConfig.getParameter<bool>         ( "addLRValues" );
  electronLRFile_   = iConfig.getParameter<std::string>  ( "electronLRFile" );
  
  // construct resolution calculator
  if(addResolutions_){
    theResoCalc_= new ObjectResolutionCalc(edm::FileInPath(electronResoFile_).fullPath(), useNNReso_);
  }

  if (iConfig.exists("isoDeposits")) {
     edm::ParameterSet depconf = iConfig.getParameter<edm::ParameterSet>("isoDeposits");
     if (depconf.exists("tracker")) isoDepositLabels_.push_back(std::make_pair(TrackerIso, depconf.getParameter<edm::InputTag>("tracker")));
     if (depconf.exists("ecal"))    isoDepositLabels_.push_back(std::make_pair(ECalIso, depconf.getParameter<edm::InputTag>("ecal")));
     if (depconf.exists("hcal"))    isoDepositLabels_.push_back(std::make_pair(HCalIso, depconf.getParameter<edm::InputTag>("hcal")));
     if (depconf.exists("user")) {
        std::vector<edm::InputTag> userdeps = depconf.getParameter<std::vector<edm::InputTag> >("user");
        std::vector<edm::InputTag>::const_iterator it = userdeps.begin(), ed = userdeps.end();
        int key = UserBaseIso;
        for ( ; it != ed; ++it, ++key) {
            isoDepositLabels_.push_back(std::make_pair(IsolationKeys(key), *it));
        }
     }
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

  if (isolator_.enabled()) isolator_.beginEvent(iEvent);

  std::vector<edm::Handle<edm::ValueMap<IsoDeposit> > > deposits(isoDepositLabels_.size());
  for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
    iEvent.getByLabel(isoDepositLabels_[j].second, deposits[j]);
  }

  // prepare the MC matching
  edm::Handle<edm::Association<reco::GenParticleCollection> > genMatch;
  if (addGenMatch_) {
    iEvent.getByLabel(genMatchSrc_, genMatch);
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

  edm::Handle<edm::View<reco::Track> > tracks;
  iEvent.getByLabel(tracksSrc_, tracks);

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
    
    // Isolation
    if (isolator_.enabled()) {
        isolator_.fill(*electrons, idx, isolatorTmpStorage_);
        typedef pat::helper::MultiIsolator::IsolationValuePairs IsolationValuePairs;
        // better to loop backwards, so the vector is resized less times
        for (IsolationValuePairs::const_reverse_iterator it = isolatorTmpStorage_.rbegin(), ed = isolatorTmpStorage_.rend(); it != ed; ++it) {
            anElectron.setIsolation(it->first, it->second);
        }
    }

    for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
        anElectron.setIsoDeposit(isoDepositLabels_[j].first, (*deposits[j])[elecsRef]);
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
  if (addLRValues_) delete theLeptonLRCalc_;

  if (isolator_.enabled()) isolator_.endEvent();

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


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATElectronProducer);
