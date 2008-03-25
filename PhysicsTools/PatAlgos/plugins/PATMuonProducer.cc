//
// $Id: PATMuonProducer.cc,v 1.1.2.1 2008/03/06 10:40:10 llista Exp $
//

#include "PhysicsTools/PatAlgos/plugins/PATMuonProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "PhysicsTools/Utilities/interface/DeltaR.h"
#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
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
  edm::Handle<reco::MuIsoDepositAssociationMap> trackerIso;
  if (doTrkIso_) {
    if (hocalIsoSrc_.label() == "famos") { // switch off for full sim, since we switched back to using muon-obj embedded info
      iEvent.getByLabel(trackIsoSrc_, trackerIso);
    }
  }
  edm::Handle<reco::MuIsoDepositAssociationMap> ecalIso;
  edm::Handle<reco::MuIsoDepositAssociationMap> hcalIso;
  edm::Handle<reco::MuIsoDepositAssociationMap> hocalIso;
  if (doCalIso_) {
    if (hocalIsoSrc_.label() == "famos") { // switch off for full sim, since we switched back to using muon-obj embedded info
      iEvent.getByLabel(ecalIsoSrc_, ecalIso);
      iEvent.getByLabel(hcalIsoSrc_, hcalIso);
      if (hocalIsoSrc_.label() != "famos") iEvent.getByLabel(hocalIsoSrc_, hocalIso);
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
    if (doTrkIso_) {
      std::pair<float, int> sumPtAndNTracks03;
      if (hocalIsoSrc_.label() != "famos") {
        // use the muon embedded muon isolation
        sumPtAndNTracks03.first = aMuon.getIsolationR03().sumPt;
        // use the muon isolation from the isolation maps (not yet stored in the 152 reco samples)
        //const reco::MuIsoDeposit & depTracker = (*trackerIso)[itMuon->combinedMuon()];
        //// cone hardcoded, corresponds to default in recent CMSSW versions
        //sumPtAndNTracks03 = depTracker.depositAndCountWithin(0.3);
      } else {
        const reco::MuIsoDeposit & depTracker = (*trackerIso)[itMuon->track()];
        // cone hardcoded, corresponds to default in recent CMSSW versions
        sumPtAndNTracks03 = depTracker.depositAndCountWithin(0.3);
      }
      aMuon.setTrackIso(sumPtAndNTracks03.first);
    }
    // do calorimeter isolation
    if (doCalIso_) {
      if (hocalIsoSrc_.label() != "famos") {
        // use the muon embedded muon isolation
        aMuon.setCaloIso(aMuon.getIsolationR03().emEt + aMuon.getIsolationR03().hadEt + aMuon.getIsolationR03().hoEt);
        // use the muon isolation from the isolation maps (not yet stored in the 152 reco samples)
        //const reco::MuIsoDeposit & depEcal = (*ecalIso)[itMuon->combinedMuon()];
        //const reco::MuIsoDeposit & depHcal = (*hcalIso)[itMuon->combinedMuon()];
        //const reco::MuIsoDeposit & depHOcal = (*hocalIso)[itMuon->combinedMuon()];
        //// cone hardcoded, corresponds to default in recent CMSSW versions
        //aMuon.setCaloIso(depEcal.depositWithin(0.3)+depHcal.depositWithin(0.3)+depHOcal.depositWithin(0.3));
      } else {
        const reco::MuIsoDeposit & depEcal = (*ecalIso)[itMuon->track()];
        const reco::MuIsoDeposit & depHcal = (*hcalIso)[itMuon->track()];
        // cone hardcoded, corresponds to default in recent CMSSW versions
        aMuon.setCaloIso(depEcal.depositWithin(0.3)+depHcal.depositWithin(0.3));
      }
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
