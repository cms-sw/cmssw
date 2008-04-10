//
// $Id: PATMuonProducer.cc,v 1.1 2008/03/06 09:23:10 llista Exp $
//

#include "PhysicsTools/PatAlgos/plugins/PATMuonProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "PhysicsTools/Utilities/interface/DeltaR.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "DataFormats/Common/interface/Association.h"

#include "PhysicsTools/PatUtils/interface/ObjectResolutionCalc.h"
#include "PhysicsTools/PatUtils/interface/LeptonLRCalc.h"

#include "TMath.h"

#include <vector>
#include <memory>


using namespace pat;


PATMuonProducer::PATMuonProducer(const edm::ParameterSet & iConfig) {

  // general configurables
  muonSrc_       = iConfig.getParameter<edm::InputTag>( "muonSource" );
  // MC matching configurables
  addGenMatch_   = iConfig.getParameter<bool>         ( "addGenMatch" );
  genPartSrc_    = iConfig.getParameter<edm::InputTag>( "genParticleMatch" );
  // resolution configurables
  addResolutions_= iConfig.getParameter<bool>         ( "addResolutions" );
  useNNReso_     = iConfig.getParameter<bool>         ( "useNNResolutions" );
  muonResoFile_  = iConfig.getParameter<std::string>  ( "muonResoFile" );
  // isolation configurables
  //! use them only if doIsoFromDeposit is true
  doIsoFromDeposit_ = iConfig.getParameter<bool>      ( "doIsoFromDeposit" );
  doTrkIso_      = iConfig.getParameter<bool>         ( "doTrkIsolation" );
  doCalIso_      = iConfig.getParameter<bool>         ( "doCalIsolation" );
  trackIsoSrc_   = iConfig.getParameter<edm::InputTag>( "trackIsoSource" );
  ecalIsoSrc_    = iConfig.getParameter<edm::InputTag>( "ecalIsoSource" );
  hcalIsoSrc_    = iConfig.getParameter<edm::InputTag>( "hcalIsoSource" );
  hocalIsoSrc_   = iConfig.getParameter<edm::InputTag>( "hocalIsoSource" );
  // muon ID configurables
  addMuonID_     = iConfig.getParameter<bool>         ( "addMuonID" );
  // likelihood ratio configurables
  addLRValues_   = iConfig.getParameter<bool>         ( "addLRValues" );
  tracksSrc_     = iConfig.getParameter<edm::InputTag>( "tracksSource" );
  muonLRFile_    = iConfig.getParameter<std::string>  ( "muonLRFile" );

  // construct resolution calculator
  if (addResolutions_) {
    theResoCalc_ = new ObjectResolutionCalc(edm::FileInPath(muonResoFile_).fullPath(), useNNReso_);
  }

  // produces vector of muons
  produces<std::vector<Muon> >();

}


PATMuonProducer::~PATMuonProducer() {
  if (addResolutions_) delete theResoCalc_;
}


void PATMuonProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {

  // Get the collection of muons from the event
  edm::Handle<edm::View<MuonType> > muons;
  iEvent.getByLabel(muonSrc_, muons);

  // prepare the MC matching
  edm::Handle<edm::Association<reco::GenParticleCollection> > genMatch;
  if (addGenMatch_) iEvent.getByLabel(genPartSrc_, genMatch);

  // prepare isolation calculation
  edm::Handle<reco::IsoDepositMap> trackerIso;
  edm::Handle<reco::IsoDepositMap> ecalIso;
  edm::Handle<reco::IsoDepositMap> hcalIso;
  edm::Handle<reco::IsoDepositMap> hocalIso;
  if (doIsoFromDeposit_){
    if (doTrkIso_) {
      iEvent.getByLabel(trackIsoSrc_, trackerIso);
    }
    if (doCalIso_) {
      iEvent.getByLabel(ecalIsoSrc_, ecalIso);
      iEvent.getByLabel(hcalIsoSrc_, hcalIso);
      iEvent.getByLabel(hocalIsoSrc_, hocalIso);
    }
  }

  // prepare LR calculation
  edm::Handle<edm::View<reco::Track> > tracks;
  if (addLRValues_) {
    iEvent.getByLabel(tracksSrc_, tracks);
    theLeptonLRCalc_ = new LeptonLRCalc(iSetup, "", edm::FileInPath(muonLRFile_).fullPath(), "");
  }

  // loop over muons
  std::vector<Muon> * patMuons = new std::vector<Muon>();
  for (edm::View<MuonType>::const_iterator itMuon = muons->begin(); itMuon != muons->end(); ++itMuon) {
    // construct the Muon from the ref -> save ref to original object
    unsigned int idx = itMuon - muons->begin();
    edm::RefToBase<MuonType> muonsRef = muons->refAt(idx);
    Muon aMuon(muonsRef);
    // match to generated final state muons
    if (addGenMatch_) {
      reco::GenParticleRef genMuon = (*genMatch)[muonsRef];
      if (genMuon.isNonnull() && genMuon.isAvailable() ) {
        aMuon.setGenLepton(*genMuon);
      } else {
        aMuon.setGenLepton(reco::Particle(0, reco::Particle::LorentzVector(0,0,0,0))); // TQAF way of setting "null"
      }
    }
    // add resolution info
    if (addResolutions_) {
      (*theResoCalc_)(aMuon);
    }
    // do tracker isolation
    if (doIsoFromDeposit_){
      if (doTrkIso_) {
	const reco::IsoDeposit & depTracker = (*trackerIso)[muonsRef];
	aMuon.setTrackIso(depTracker.depositWithin(0.3));
      }
      // do calorimeter isolation
      if (doCalIso_) {
	const reco::IsoDeposit & depEcal = (*ecalIso)[muonsRef];
	const reco::IsoDeposit & depHcal = (*hcalIso)[muonsRef];
	const reco::IsoDeposit & depHOcal = (*hocalIso)[muonsRef];
	
	//! take a sumEt in th ehardcoded cone of size 0.3
	double sumEtCal = depEcal.depositWithin(0.3);
	sumEtCal += depHcal.depositWithin(0.3);
	sumEtCal += depHOcal.depositWithin(0.3);
	aMuon.setCaloIso(sumEtCal);
      }
    } else { // pick from the muon itself : duplicate data here, since it's all available in the muon itself
      aMuon.setTrackIso(aMuon.isolationR03().sumPt);
      aMuon.setCaloIso(aMuon.isolationR03().emEt + aMuon.isolationR03().hadEt + aMuon.isolationR03().hoEt);
    }

    // add muon ID info
    if (addMuonID_) {
      //      aMuon.setLeptonID((float) TMath::Prob((Float_t) itMuon->combinedMuon()->chi2(), (Int_t) itMuon->combinedMuon()->ndof()));
      // no combinedMuon in fastsim
      aMuon.setLeptonID((float) TMath::Prob((Float_t) itMuon->track()->chi2(), (Int_t) itMuon->track()->ndof()));
    }
    // add lepton LR info
    if (addLRValues_) {
      theLeptonLRCalc_->calcLikelihood(aMuon, tracks, iEvent);
    }
    // add sel to selected
    patMuons->push_back(aMuon);
  }

  // sort muons in pt
  std::sort(patMuons->begin(), patMuons->end(), pTComparator_);

  // put genEvt object in Event
  std::auto_ptr<std::vector<Muon> > ptr(patMuons);
  iEvent.put(ptr);

  if (addLRValues_) delete theLeptonLRCalc_;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATMuonProducer);
