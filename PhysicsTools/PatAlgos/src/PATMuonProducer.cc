//
// $Id$
//

#include "PhysicsTools/PatAlgos/interface/PATMuonProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "PhysicsTools/Utilities/interface/DeltaR.h"
#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

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
  genPartSrc_    = iConfig.getParameter<edm::InputTag>( "genParticleSource" );
  maxDeltaR_     = iConfig.getParameter<double>       ( "maxDeltaR" );
  minRecoOnGenEt_= iConfig.getParameter<double>       ( "minRecoOnGenEt" );
  maxRecoOnGenEt_= iConfig.getParameter<double>       ( "maxRecoOnGenEt" );
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
  edm::Handle<std::vector<MuonType> > muonsHandle;
  iEvent.getByLabel(muonSrc_, muonsHandle);
  std::vector<MuonType> muons = *muonsHandle;

  // prepare the MC matching
  edm::Handle<reco::CandidateCollection> particles;
  if (addGenMatch_) {
    iEvent.getByLabel(genPartSrc_, particles);
    matchTruth(*particles, muons);
  }

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
  edm::Handle<reco::TrackCollection> trackHandle;
  if (addLRValues_) {
    iEvent.getByLabel(tracksSrc_, trackHandle);
    theLeptonLRCalc_ = new LeptonLRCalc(iSetup, "", edm::FileInPath(muonLRFile_).fullPath(), "");
  }

  // loop over muons
  std::vector<Muon> * patMuons = new std::vector<Muon>();
  for (size_t m = 0; m < muons.size(); ++m) {
    // construct the muon
    Muon aMuon(muons[m]);
    // match to generated final state muons
    if (addGenMatch_) {
      aMuon.setGenLepton(findTruth(*particles, muons[m]));
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
        //const reco::MuIsoDeposit & depTracker = (*trackerIso)[muons[m].combinedMuon()];
        //// cone hardcoded, corresponds to default in recent CMSSW versions
        //sumPtAndNTracks03 = depTracker.depositAndCountWithin(0.3);
      } else {
        const reco::MuIsoDeposit & depTracker = (*trackerIso)[muons[m].track()];
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
        //const reco::MuIsoDeposit & depEcal = (*ecalIso)[muons[m].combinedMuon()];
        //const reco::MuIsoDeposit & depHcal = (*hcalIso)[muons[m].combinedMuon()];
        //const reco::MuIsoDeposit & depHOcal = (*hocalIso)[muons[m].combinedMuon()];
        //// cone hardcoded, corresponds to default in recent CMSSW versions
        //aMuon.setCaloIso(depEcal.depositWithin(0.3)+depHcal.depositWithin(0.3)+depHOcal.depositWithin(0.3));
      } else {
        const reco::MuIsoDeposit & depEcal = (*ecalIso)[muons[m].track()];
        const reco::MuIsoDeposit & depHcal = (*hcalIso)[muons[m].track()];
        // cone hardcoded, corresponds to default in recent CMSSW versions
        aMuon.setCaloIso(depEcal.depositWithin(0.3)+depHcal.depositWithin(0.3));
      }
    }
    // add muon ID info
    if (addMuonID_) {
//      aMuon.setLeptonID((float) TMath::Prob((Float_t) muons[m].combinedMuon()->chi2(), (Int_t) muons[m].combinedMuon()->ndof()));
// no combinedMuon in fastsim
      aMuon.setLeptonID((float) TMath::Prob((Float_t) muons[m].track()->chi2(), (Int_t) muons[m].track()->ndof()));
    }
    // add lepton LR info
    if (addLRValues_) {
      theLeptonLRCalc_->calcLikelihood(aMuon, trackHandle, iEvent);
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


reco::GenParticleCandidate PATMuonProducer::findTruth(const reco::CandidateCollection & parts, const MuonType & muon) {
  reco::GenParticleCandidate gen;
  for(unsigned int idx=0; idx!=pairGenRecoMuonsVector_.size(); ++idx){
    std::pair<const reco::Candidate*, MuonType*> pairGenRecoMuons;
    pairGenRecoMuons = pairGenRecoMuonsVector_[idx];
    float dR = DeltaR<reco::Candidate>()( muon, *(pairGenRecoMuons.second));
    if( !(dR > 0) ){
      gen = *(dynamic_cast<const reco::GenParticleCandidate*>( pairGenRecoMuons.first ) );
    }
  }
  return gen;
}


void PATMuonProducer::matchTruth(const reco::CandidateCollection & parts, std::vector<MuonType> & muons) {
  pairGenRecoMuonsVector_.clear();
  reco::CandidateCollection::const_iterator part = parts.begin();
  for( ; part != parts.end(); ++part ){
    reco::GenParticleCandidate gen = *(dynamic_cast<const reco::GenParticleCandidate*>( &(*part)) );
    if( abs(gen.pdgId())==13 && gen.status()==1 ){
      bool  found = false;
      float minDR = 99999;
      MuonType* rec = 0;
      MuonTypeCollection::iterator muon = muons.begin();
      for ( ; muon !=muons.end(); ++muon){
	float dR = DeltaR<reco::Candidate>()( gen, *muon);
	float ptRecOverGen = muon->pt()/gen.pt();
	if ( ( ptRecOverGen > minRecoOnGenEt_ ) && 
	     ( ptRecOverGen < maxRecoOnGenEt_ ) && 
	     ( dR < maxDeltaR_) ){
	  if ( dR < minDR ){
	    rec = &(*muon);
	    minDR = dR;
	    found = true;
	  }
	}
      }
      if( found == true ){
	pairGenRecoMuonsVector_.push_back( std::pair<const reco::Candidate*,MuonType*>(&(*part), rec ) );
      }
    }
  }
}
