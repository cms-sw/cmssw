//
// $Id: Tau.cc,v 1.12 2008/11/28 19:02:15 lowette Exp $
//

#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/JetReco/interface/GenJet.h"


using namespace pat;


/// default constructor
Tau::Tau() :
    Lepton<reco::BaseTau>(),
    embeddedIsolationTracks_(false),
    embeddedLeadTrack_(false),
    embeddedSignalTracks_(false)
{
}


/// constructor from reco::BaseTau
Tau::Tau(const reco::BaseTau & aTau) :
    Lepton<reco::BaseTau>(aTau),
    embeddedIsolationTracks_(false),
    embeddedLeadTrack_(false),
    embeddedSignalTracks_(false)
{
    const reco::PFTau * pfTau = dynamic_cast<const reco::PFTau *>(&aTau);
    if (pfTau != 0) pfSpecific_.push_back(pat::tau::TauPFSpecific(*pfTau));
    const reco::CaloTau * caloTau = dynamic_cast<const reco::CaloTau *>(&aTau);
    if (caloTau != 0) caloSpecific_.push_back(pat::tau::TauCaloSpecific(*caloTau));
}


/// constructor from ref to reco::BaseTau
Tau::Tau(const edm::RefToBase<reco::BaseTau> & aTauRef) :
    Lepton<reco::BaseTau>(aTauRef),
    embeddedIsolationTracks_(false),
    embeddedLeadTrack_(false),
    embeddedSignalTracks_(false)
{
    const reco::PFTau * pfTau = dynamic_cast<const reco::PFTau *>(aTauRef.get());
    if (pfTau != 0) pfSpecific_.push_back(pat::tau::TauPFSpecific(*pfTau));
    const reco::CaloTau * caloTau = dynamic_cast<const reco::CaloTau *>(aTauRef.get());
    if (caloTau != 0) caloSpecific_.push_back(pat::tau::TauCaloSpecific(*caloTau));
}

/// constructor from ref to reco::BaseTau
Tau::Tau(const edm::Ptr<reco::BaseTau> & aTauRef) :
    Lepton<reco::BaseTau>(aTauRef),
    embeddedIsolationTracks_(false),
    embeddedLeadTrack_(false),
    embeddedSignalTracks_(false)
{
    const reco::PFTau * pfTau = dynamic_cast<const reco::PFTau *>(aTauRef.get());
    if (pfTau != 0) pfSpecific_.push_back(pat::tau::TauPFSpecific(*pfTau));
    const reco::CaloTau * caloTau = dynamic_cast<const reco::CaloTau *>(aTauRef.get());
    if (caloTau != 0) caloSpecific_.push_back(pat::tau::TauCaloSpecific(*caloTau));
}



/// destructor
Tau::~Tau() {
}


/// override the reco::BaseTau::isolationTracks method, to access the internal storage of the track
const reco::TrackRefVector & Tau::isolationTracks() const {
  if (embeddedIsolationTracks_) {
    if (!isolationTracksTransientRefVectorFixed_) {
        reco::TrackRefVector trackRefVec;
        for (unsigned int i = 0; i < isolationTracks_.size(); i++) {
          trackRefVec.push_back(reco::TrackRef(&isolationTracks_, i));
        }
        isolationTracksTransientRefVector_.swap(trackRefVec);
        isolationTracksTransientRefVectorFixed_ = true;
    }
    return isolationTracksTransientRefVector_;
  } else {
    return reco::BaseTau::isolationTracks();
  }
}


/// override the reco::BaseTau::track method, to access the internal storage of the track
reco::TrackRef Tau::leadTrack() const {
  if (embeddedLeadTrack_) {
    return reco::TrackRef(&leadTrack_, 0);
  } else {
    return reco::BaseTau::leadTrack();
  }
}


/// override the reco::BaseTau::track method, to access the internal storage of the track
const reco::TrackRefVector & Tau::signalTracks() const {
  if (embeddedSignalTracks_) {
    reco::TrackRefVector trackRefVec;
    if (!signalTracksTransientRefVectorFixed_) {
        for (unsigned int i = 0; i < signalTracks_.size(); i++) {
          trackRefVec.push_back(reco::TrackRef(&signalTracks_, i));
        }
        signalTracksTransientRefVector_.swap(trackRefVec);
        signalTracksTransientRefVectorFixed_ = true;
    }
    return signalTracksTransientRefVector_;
  } else {
    return reco::BaseTau::signalTracks();
  }
}


/// method to store the isolation tracks internally
void Tau::embedIsolationTracks() {
  isolationTracks_.clear();
  reco::TrackRefVector trackRefVec = reco::BaseTau::isolationTracks();
  for (unsigned int i = 0; i < trackRefVec.size(); i++) {
    isolationTracks_.push_back(*trackRefVec.at(i));
  }
  embeddedIsolationTracks_ = true;
}


/// method to store the isolation tracks internally
void Tau::embedLeadTrack() {
  leadTrack_.clear();
  if (reco::BaseTau::leadTrack().isNonnull()) {
      leadTrack_.push_back(*reco::BaseTau::leadTrack());
      embeddedLeadTrack_ = true;
  }
}


/// method to store the isolation tracks internally
void Tau::embedSignalTracks(){
  signalTracks_.clear();
  reco::TrackRefVector trackRefVec = reco::BaseTau::signalTracks();
  for (unsigned int i = 0; i < trackRefVec.size(); i++) {
    signalTracks_.push_back(*trackRefVec.at(i));
  }
  embeddedSignalTracks_ = true;
}


/// method to set the matched generated jet
void Tau::setGenJet(const reco::GenJetRef& gj) {
  genJet_.clear();
  genJet_.push_back(*gj);
}

/// return the matched generated jet
const reco::GenJet * Tau::genJet() const {
  return (genJet_.size() > 0 ? &genJet_.front() : 0);
}


// method to retrieve a tau ID (or throw)
float Tau::tauID(const std::string & name) const {
  for (std::vector<IdPair>::const_iterator it = tauIDs_.begin(), ed = tauIDs_.end(); it != ed; ++it) {
    if (it->first == name) return it->second;
  }
  cms::Exception ex("Key not found");
  ex << "pat::Tau: the ID " << name << " can't be found in this pat::Tau.\n";
  ex << "The available IDs are: ";
  for (std::vector<IdPair>::const_iterator it = tauIDs_.begin(), ed = tauIDs_.end(); it != ed; ++it) {
    ex << "'" << it->first << "' ";
  }
  ex << ".\n";
  throw ex;
}
// check if an ID is there
bool Tau::isTauIDAvailable(const std::string & name) const {
  for (std::vector<IdPair>::const_iterator it = tauIDs_.begin(), ed = tauIDs_.end(); it != ed; ++it) {
    if (it->first == name) return true;
  }
  return false;
}


const pat::tau::TauPFSpecific & Tau::pfSpecific() const {
  if (!isPFTau()) throw cms::Exception("Type Error") << "Requesting a PFTau-specific information from a pat::Tau which wasn't made from a PFTau.\n";
  return pfSpecific_[0]; 
}

const pat::tau::TauCaloSpecific & Tau::caloSpecific() const {
  if (!isCaloTau()) throw cms::Exception("Type Error") << "Requesting a CaloTau-specific information from a pat::Tau which wasn't made from a CaloTau.\n";
  return caloSpecific_[0]; 
}
