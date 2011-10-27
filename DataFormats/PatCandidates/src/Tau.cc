//
// $Id: Tau.cc,v 1.21 2011/09/29 16:34:57 veelken Exp $
//

#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/JetReco/interface/GenJet.h"


using namespace pat;


/// default constructor
Tau::Tau() :
    Lepton<reco::BaseTau>()
    ,embeddedIsolationTracks_(false)
    ,isolationTracksTransientRefVectorFixed_(false)
    ,embeddedLeadTrack_(false)
    ,embeddedSignalTracks_(false)
    ,signalTracksTransientRefVectorFixed_(false)
    ,embeddedLeadPFCand_(false)
    ,embeddedLeadPFChargedHadrCand_(false)
    ,embeddedLeadPFNeutralCand_(false)
    ,embeddedSignalPFCands_(false)
    ,signalPFCandsRefVectorFixed_(false)
    ,embeddedSignalPFChargedHadrCands_(false)
    ,signalPFChargedHadrCandsRefVectorFixed_(false)
    ,embeddedSignalPFNeutralHadrCands_(false)
    ,signalPFNeutralHadrCandsRefVectorFixed_(false)
    ,embeddedSignalPFGammaCands_(false)
    ,signalPFGammaCandsRefVectorFixed_(false)
    ,embeddedIsolationPFCands_(false)
    ,isolationPFCandsRefVectorFixed_(false)
    ,embeddedIsolationPFChargedHadrCands_(false)
    ,isolationPFChargedHadrCandsRefVectorFixed_(false)
    ,embeddedIsolationPFNeutralHadrCands_(false)
    ,isolationPFNeutralHadrCandsRefVectorFixed_(false)
    ,embeddedIsolationPFGammaCands_(false)
    ,isolationPFGammaCandsRefVectorFixed_(false)
{
}

/// constructor from reco::BaseTau
Tau::Tau(const reco::BaseTau & aTau) :
    Lepton<reco::BaseTau>(aTau)
    ,embeddedIsolationTracks_(false)
    ,isolationTracksTransientRefVectorFixed_(false)
    ,embeddedLeadTrack_(false)
    ,embeddedSignalTracks_(false)
    ,signalTracksTransientRefVectorFixed_(false)
    ,embeddedLeadPFCand_(false)
    ,embeddedLeadPFChargedHadrCand_(false)
    ,embeddedLeadPFNeutralCand_(false)
    ,embeddedSignalPFCands_(false)
    ,signalPFCandsRefVectorFixed_(false)
    ,embeddedSignalPFChargedHadrCands_(false)
    ,signalPFChargedHadrCandsRefVectorFixed_(false)
    ,embeddedSignalPFNeutralHadrCands_(false)
    ,signalPFNeutralHadrCandsRefVectorFixed_(false)
    ,embeddedSignalPFGammaCands_(false)
    ,signalPFGammaCandsRefVectorFixed_(false)
    ,embeddedIsolationPFCands_(false)
    ,isolationPFCandsRefVectorFixed_(false)
    ,embeddedIsolationPFChargedHadrCands_(false)
    ,isolationPFChargedHadrCandsRefVectorFixed_(false)
    ,embeddedIsolationPFNeutralHadrCands_(false)
    ,isolationPFNeutralHadrCandsRefVectorFixed_(false)
    ,embeddedIsolationPFGammaCands_(false)
    ,isolationPFGammaCandsRefVectorFixed_(false)
{
    const reco::PFTau * pfTau = dynamic_cast<const reco::PFTau *>(&aTau);
    if (pfTau != 0) pfSpecific_.push_back(pat::tau::TauPFSpecific(*pfTau));
    const reco::CaloTau * caloTau = dynamic_cast<const reco::CaloTau *>(&aTau);
    if (caloTau != 0) caloSpecific_.push_back(pat::tau::TauCaloSpecific(*caloTau));
}

/// constructor from ref to reco::BaseTau
Tau::Tau(const edm::RefToBase<reco::BaseTau> & aTauRef) :
    Lepton<reco::BaseTau>(aTauRef)
    ,embeddedIsolationTracks_(false)
    ,isolationTracksTransientRefVectorFixed_(false)
    ,embeddedLeadTrack_(false)
    ,embeddedSignalTracks_(false)
    ,signalTracksTransientRefVectorFixed_(false)
    ,embeddedLeadPFCand_(false)
    ,embeddedLeadPFChargedHadrCand_(false)
    ,embeddedLeadPFNeutralCand_(false)
    ,embeddedSignalPFCands_(false)
    ,signalPFCandsRefVectorFixed_(false)
    ,embeddedSignalPFChargedHadrCands_(false)
    ,signalPFChargedHadrCandsRefVectorFixed_(false)
    ,embeddedSignalPFNeutralHadrCands_(false)
    ,signalPFNeutralHadrCandsRefVectorFixed_(false)
    ,embeddedSignalPFGammaCands_(false)
    ,signalPFGammaCandsRefVectorFixed_(false)
    ,embeddedIsolationPFCands_(false)
    ,isolationPFCandsRefVectorFixed_(false)
    ,embeddedIsolationPFChargedHadrCands_(false)
    ,isolationPFChargedHadrCandsRefVectorFixed_(false)
    ,embeddedIsolationPFNeutralHadrCands_(false)
    ,isolationPFNeutralHadrCandsRefVectorFixed_(false)
    ,embeddedIsolationPFGammaCands_(false)
    ,isolationPFGammaCandsRefVectorFixed_(false)
{
    const reco::PFTau * pfTau = dynamic_cast<const reco::PFTau *>(aTauRef.get());
    if (pfTau != 0) pfSpecific_.push_back(pat::tau::TauPFSpecific(*pfTau));
    const reco::CaloTau * caloTau = dynamic_cast<const reco::CaloTau *>(aTauRef.get());
    if (caloTau != 0) caloSpecific_.push_back(pat::tau::TauCaloSpecific(*caloTau));
}

/// constructor from ref to reco::BaseTau
Tau::Tau(const edm::Ptr<reco::BaseTau> & aTauRef) :
    Lepton<reco::BaseTau>(aTauRef)
    ,embeddedIsolationTracks_(false)
    ,isolationTracksTransientRefVectorFixed_(false)
    ,embeddedLeadTrack_(false)
    ,embeddedSignalTracks_(false)
    ,signalTracksTransientRefVectorFixed_(false)
    ,embeddedLeadPFCand_(false)
    ,embeddedLeadPFChargedHadrCand_(false)
    ,embeddedLeadPFNeutralCand_(false)
    ,embeddedSignalPFCands_(false)
    ,signalPFCandsRefVectorFixed_(false)
    ,embeddedSignalPFChargedHadrCands_(false)
    ,signalPFChargedHadrCandsRefVectorFixed_(false)
    ,embeddedSignalPFNeutralHadrCands_(false)
    ,signalPFNeutralHadrCandsRefVectorFixed_(false)
    ,embeddedSignalPFGammaCands_(false)
    ,signalPFGammaCandsRefVectorFixed_(false)
    ,embeddedIsolationPFCands_(false)
    ,isolationPFCandsRefVectorFixed_(false)
    ,embeddedIsolationPFChargedHadrCands_(false)
    ,isolationPFChargedHadrCandsRefVectorFixed_(false)
    ,embeddedIsolationPFNeutralHadrCands_(false)
    ,isolationPFNeutralHadrCandsRefVectorFixed_(false)
    ,embeddedIsolationPFGammaCands_(false)
    ,isolationPFGammaCandsRefVectorFixed_(false)
{
    const reco::PFTau * pfTau = dynamic_cast<const reco::PFTau *>(aTauRef.get());
    if (pfTau != 0) pfSpecific_.push_back(pat::tau::TauPFSpecific(*pfTau));
    const reco::CaloTau * caloTau = dynamic_cast<const reco::CaloTau *>(aTauRef.get());
    if (caloTau != 0) caloSpecific_.push_back(pat::tau::TauCaloSpecific(*caloTau));
}

/// destructor
Tau::~Tau() {
}

std::ostream& 
reco::operator<<(std::ostream& out, const pat::Tau& obj) 
{
  if(!out) return out;
  
  out << "\tpat::Tau: ";
  out << std::setiosflags(std::ios::right);
  out << std::setiosflags(std::ios::fixed);
  out << std::setprecision(3);
  out << " E/pT/eta/phi " 
      << obj.energy()<<"/"
      << obj.pt()<<"/"
      << obj.eta()<<"/"
      << obj.phi();
  return out; 
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

const reco::Candidate::LorentzVector& Tau::p4Jet() const
{
  if ( isCaloTau() ) return caloSpecific().p4Jet_;
  if ( isPFTau()   ) return pfSpecific().p4Jet_;
  throw cms::Exception("Type Error") << "Requesting a CaloTau/PFTau-specific information from a pat::Tau which wasn't made from either a CaloTau or a PFTau.\n";
}

float Tau::etaetaMoment() const
{
  if ( isCaloTau() ) return caloSpecific().etaetaMoment_;
  if ( isPFTau()   ) return pfSpecific().etaetaMoment_;
  throw cms::Exception("Type Error") << "Requesting a CaloTau/PFTau-specific information from a pat::Tau which wasn't made from either a CaloTau or a PFTau.\n";
}

float Tau::phiphiMoment() const
{
  if ( isCaloTau() ) return caloSpecific().phiphiMoment_;
  if ( isPFTau()   ) return pfSpecific().phiphiMoment_;
  throw cms::Exception("Type Error") << "Requesting a CaloTau/PFTau-specific information from a pat::Tau which wasn't made from either a CaloTau or a PFTau.\n";
}

float Tau::etaphiMoment() const
{
  if ( isCaloTau() ) return caloSpecific().etaphiMoment_;
  if ( isPFTau()   ) return pfSpecific().etaphiMoment_;
  throw cms::Exception("Type Error") << "Requesting a CaloTau/PFTau-specific information from a pat::Tau which wasn't made from either a CaloTau or a PFTau.\n";
}

void Tau::setDecayMode(int decayMode)
{
  if (!isPFTau()) throw cms::Exception("Type Error") << "Requesting a PFTau-specific information from a pat::Tau which wasn't made from a PFTau.\n";
  pfSpecific_[0].decayMode_ = decayMode;
} 

/// method to store the leading candidate internally
void Tau::embedLeadPFCand() {
  if (!isPFTau() ) {//additional check with warning in pat::tau producer
    return;
  }
  leadPFCand_.clear();
  if (pfSpecific_[0].leadPFCand_.isNonnull() ) {
    leadPFCand_.push_back(*pfSpecific_[0].leadPFCand_); //already set in C-tor
    embeddedLeadPFCand_ = true;
  }
}
/// method to store the leading candidate internally
void Tau::embedLeadPFChargedHadrCand() {
  if (!isPFTau() ) {//additional check with warning in pat::tau producer
    return;
  }
  leadPFChargedHadrCand_.clear();
  if (pfSpecific_[0].leadPFChargedHadrCand_.isNonnull() ) {
    leadPFChargedHadrCand_.push_back(*pfSpecific_[0].leadPFChargedHadrCand_); //already set in C-tor
    embeddedLeadPFChargedHadrCand_ = true;
  }
}
/// method to store the leading candidate internally
void Tau::embedLeadPFNeutralCand() {
  if (!isPFTau() ) {//additional check with warning in pat::tau producer
    return;
  }
  leadPFNeutralCand_.clear();
  if (pfSpecific_[0].leadPFNeutralCand_.isNonnull() ) {
    leadPFNeutralCand_.push_back(*pfSpecific_[0].leadPFNeutralCand_); //already set in C-tor
    embeddedLeadPFNeutralCand_ = true;
  }
}

void Tau::embedSignalPFCands() {
  if (!isPFTau() ) {//additional check with warning in pat::tau producer
    return;
  }
  reco::PFCandidateRefVector candRefVec = pfSpecific_[0].selectedSignalPFCands_;
  for (unsigned int i = 0; i < candRefVec.size(); i++) {
    signalPFCands_.push_back(*candRefVec.at(i));
  }
  embeddedSignalPFCands_ = true;
}
void Tau::embedSignalPFChargedHadrCands() {
  if (!isPFTau() ) {//additional check with warning in pat::tau producer
    return;
  }
  reco::PFCandidateRefVector candRefVec = pfSpecific_[0].selectedSignalPFChargedHadrCands_;
  for (unsigned int i = 0; i < candRefVec.size(); i++) {
    signalPFChargedHadrCands_.push_back(*candRefVec.at(i));
  }
  embeddedSignalPFChargedHadrCands_ = true;
}
void Tau::embedSignalPFNeutralHadrCands() {
  if (!isPFTau() ) {//additional check with warning in pat::tau producer
    return;
  }
  reco::PFCandidateRefVector candRefVec = pfSpecific_[0].selectedSignalPFNeutrHadrCands_;
  for (unsigned int i = 0; i < candRefVec.size(); i++) {
    signalPFNeutralHadrCands_.push_back(*candRefVec.at(i));
  }
  embeddedSignalPFNeutralHadrCands_ = true;
}
void Tau::embedSignalPFGammaCands() {
  if (!isPFTau() ) {//additional check with warning in pat::tau producer
    return;
  }
  reco::PFCandidateRefVector candRefVec = pfSpecific_[0].selectedSignalPFGammaCands_;
  for (unsigned int i = 0; i < candRefVec.size(); i++) {
    signalPFGammaCands_.push_back(*candRefVec.at(i));
  }
  embeddedSignalPFGammaCands_ = true;
}

void Tau::embedIsolationPFCands() {
  if (!isPFTau() ) {//additional check with warning in pat::tau producer
    return;
  }
  reco::PFCandidateRefVector candRefVec = pfSpecific_[0].selectedIsolationPFCands_;
  for (unsigned int i = 0; i < candRefVec.size(); i++) {
    isolationPFCands_.push_back(*candRefVec.at(i));
  }
  embeddedIsolationPFCands_ = true;
}

void Tau::embedIsolationPFChargedHadrCands() {
  if (!isPFTau() ) {//additional check with warning in pat::tau producer
    return;
  }
  reco::PFCandidateRefVector candRefVec = pfSpecific_[0].selectedIsolationPFChargedHadrCands_;
  for (unsigned int i = 0; i < candRefVec.size(); i++) {
    isolationPFChargedHadrCands_.push_back(*candRefVec.at(i));
  }
  embeddedIsolationPFChargedHadrCands_ = true;
}
void Tau::embedIsolationPFNeutralHadrCands() {
  if (!isPFTau() ) {//additional check with warning in pat::tau producer
    return;
  }
  reco::PFCandidateRefVector candRefVec = pfSpecific_[0].selectedIsolationPFNeutrHadrCands_;
  for (unsigned int i = 0; i < candRefVec.size(); i++) {
    isolationPFNeutralHadrCands_.push_back(*candRefVec.at(i));
  }
  embeddedIsolationPFNeutralHadrCands_ = true;
}
void Tau::embedIsolationPFGammaCands() {
  if (!isPFTau() ) {//additional check with warning in pat::tau producer
    return;
  }
  reco::PFCandidateRefVector candRefVec = pfSpecific_[0].selectedIsolationPFGammaCands_;
  for (unsigned int i = 0; i < candRefVec.size(); i++) {
    isolationPFGammaCands_.push_back(*candRefVec.at(i));
  }
  embeddedIsolationPFGammaCands_ = true;
}

const reco::PFCandidateRef Tau::leadPFChargedHadrCand() const { 
  if(!embeddedLeadPFChargedHadrCand_)
    return pfSpecific().leadPFChargedHadrCand_; 
  else
    return reco::PFCandidateRef(&leadPFChargedHadrCand_,0);
}

const reco::PFCandidateRef Tau::leadPFNeutralCand() const { 
  if(!embeddedLeadPFNeutralCand_)
    return pfSpecific().leadPFNeutralCand_;
  else
    return reco::PFCandidateRef(&leadPFNeutralCand_,0);
}

const reco::PFCandidateRef Tau::leadPFCand() const { 
  if(!embeddedLeadPFCand_)
    return pfSpecific().leadPFCand_;
  else
    return reco::PFCandidateRef(&leadPFCand_,0);
}

const reco::PFCandidateRefVector & Tau::signalPFCands() const { 
  if (embeddedSignalPFCands_) {
    if (!signalPFCandsRefVectorFixed_) {
      reco::PFCandidateRefVector aRefVec;
      for (unsigned int i = 0; i < signalPFCands_.size(); i++) {
	aRefVec.push_back(reco::PFCandidateRef(&signalPFCands_, i) );
      }
      signalPFCandsTransientRefVector_.swap(aRefVec);
      signalPFCandsRefVectorFixed_ = true;
    }
    return signalPFCandsTransientRefVector_;
  } else
    return pfSpecific().selectedSignalPFCands_; 
}

const reco::PFCandidateRefVector & Tau::signalPFChargedHadrCands() const {
  if (embeddedSignalPFChargedHadrCands_) {
    if (!signalPFChargedHadrCandsRefVectorFixed_) {
      reco::PFCandidateRefVector aRefVec;
      for (unsigned int i = 0; i < signalPFChargedHadrCands_.size(); i++) {
	aRefVec.push_back(reco::PFCandidateRef(&signalPFChargedHadrCands_, i) );
      }
      signalPFChargedHadrCandsTransientRefVector_.swap(aRefVec);
      signalPFChargedHadrCandsRefVectorFixed_ = true;
    }
    return signalPFChargedHadrCandsTransientRefVector_;
  } else
    return pfSpecific().selectedSignalPFChargedHadrCands_;
} 

const reco::PFCandidateRefVector & Tau::signalPFNeutrHadrCands() const {
  if (embeddedSignalPFNeutralHadrCands_) {
    if (!signalPFNeutralHadrCandsRefVectorFixed_) {
      reco::PFCandidateRefVector aRefVec;
      for (unsigned int i = 0; i < signalPFNeutralHadrCands_.size(); i++) {
	aRefVec.push_back(reco::PFCandidateRef(&signalPFNeutralHadrCands_, i) );
      }
      signalPFNeutralHadrCandsTransientRefVector_.swap(aRefVec);
      signalPFNeutralHadrCandsRefVectorFixed_ = true;
    }
    return signalPFNeutralHadrCandsTransientRefVector_;
  } else
    return pfSpecific().selectedSignalPFNeutrHadrCands_;
} 

const reco::PFCandidateRefVector & Tau::signalPFGammaCands() const {
  if (embeddedSignalPFGammaCands_) {
    if (!signalPFGammaCandsRefVectorFixed_) {
      reco::PFCandidateRefVector aRefVec;
      for (unsigned int i = 0; i < signalPFGammaCands_.size(); i++) {
	aRefVec.push_back(reco::PFCandidateRef(&signalPFGammaCands_, i) );
      }
      signalPFGammaCandsTransientRefVector_.swap(aRefVec);
      signalPFGammaCandsRefVectorFixed_ = true;
    }
    return signalPFGammaCandsTransientRefVector_;
  } else
    return pfSpecific().selectedSignalPFGammaCands_;
}

const std::vector<reco::RecoTauPiZero> & Tau::signalPiZeroCandidates() const {
  return pfSpecific().signalPiZeroCandidates_;
}

const reco::PFCandidateRefVector & Tau::isolationPFCands() const {
  if (embeddedIsolationPFCands_) {
    if (!isolationPFCandsRefVectorFixed_) {
      reco::PFCandidateRefVector aRefVec;
      for (unsigned int i = 0; i < isolationPFCands_.size(); i++) {
	aRefVec.push_back(reco::PFCandidateRef(&isolationPFCands_, i) );
      }
      isolationPFCandsTransientRefVector_.swap(aRefVec);
      isolationPFCandsRefVectorFixed_ = true;
    }
    return isolationPFCandsTransientRefVector_;
  } else
    return pfSpecific().selectedIsolationPFCands_;
} 

const reco::PFCandidateRefVector & Tau::isolationPFChargedHadrCands() const {
  if (embeddedIsolationPFChargedHadrCands_) {
    if (!isolationPFChargedHadrCandsRefVectorFixed_) {
      reco::PFCandidateRefVector aRefVec;
      for (unsigned int i = 0; i < isolationPFChargedHadrCands_.size(); i++) {
	aRefVec.push_back(reco::PFCandidateRef(&isolationPFChargedHadrCands_, i) );
      }
      isolationPFChargedHadrCandsTransientRefVector_.swap(aRefVec);
      isolationPFChargedHadrCandsRefVectorFixed_ = true;
    }
    return isolationPFChargedHadrCandsTransientRefVector_;
  } else
    return pfSpecific().selectedIsolationPFChargedHadrCands_;
} 

const reco::PFCandidateRefVector & Tau::isolationPFNeutrHadrCands() const {
  if (embeddedIsolationPFNeutralHadrCands_) {
    if (!isolationPFNeutralHadrCandsRefVectorFixed_) {
      reco::PFCandidateRefVector aRefVec;
      for (unsigned int i = 0; i < isolationPFNeutralHadrCands_.size(); i++) {
	aRefVec.push_back(reco::PFCandidateRef(&isolationPFNeutralHadrCands_, i) );
      }
      isolationPFNeutralHadrCandsTransientRefVector_.swap(aRefVec);
      isolationPFNeutralHadrCandsRefVectorFixed_ = true;
    }
    return isolationPFNeutralHadrCandsTransientRefVector_;
  } else
    return pfSpecific().selectedIsolationPFNeutrHadrCands_;
} 

const reco::PFCandidateRefVector & Tau::isolationPFGammaCands() const {
  if (embeddedIsolationPFGammaCands_) {
    if (!isolationPFGammaCandsRefVectorFixed_) {
      reco::PFCandidateRefVector aRefVec;
      for (unsigned int i = 0; i < isolationPFGammaCands_.size(); i++) {
	aRefVec.push_back(reco::PFCandidateRef(&isolationPFGammaCands_, i) );
      }
      isolationPFGammaCandsTransientRefVector_.swap(aRefVec);
      isolationPFGammaCandsRefVectorFixed_ = true;
    }
    return isolationPFGammaCandsTransientRefVector_;
  } else
    return pfSpecific().selectedIsolationPFGammaCands_;
}

const std::vector<reco::RecoTauPiZero> & Tau::isolationPiZeroCandidates() const {
  return pfSpecific().isolationPiZeroCandidates_;
}

/// ============= -Tau-jet Energy Correction methods ============
/// (copied from DataFormats/PatCandidates/src/Jet.cc)

// initialize the jet to a given JEC level during creation starting from Uncorrected
void Tau::initializeJEC(unsigned int level, unsigned int set)
{
  currentJECSet(set);
  currentJECLevel(level);
  setP4(jec_[set].correction(level)*p4());
}

/// return true if this jet carries the jet correction factors of a different set, for systematic studies
int Tau::jecSet(const std::string& set) const
{
  for ( std::vector<pat::TauJetCorrFactors>::const_iterator corrFactor = jec_.begin(); 
	corrFactor != jec_.end(); ++corrFactor ) {
    if ( corrFactor->jecSet() == set ) return corrFactor-jec_.begin(); 
  }
  return -1;
}

/// all available label-names of all sets of jet energy corrections
const std::vector<std::string> Tau::availableJECSets() const
{
  std::vector<std::string> sets;
  for ( std::vector<pat::TauJetCorrFactors>::const_iterator corrFactor = jec_.begin(); 
	corrFactor != jec_.end(); ++corrFactor ) {
    sets.push_back(corrFactor->jecSet());
  }
  return sets;
}

const std::vector<std::string> Tau::availableJECLevels(const int& set) const
{
  return set>=0 ? jec_.at(set).correctionLabels() : std::vector<std::string>();
}

/// correction factor to the given level for a specific set
/// of correction factors, starting from the current level
float Tau::jecFactor(const std::string& level, const std::string& set) const
{
  for ( unsigned int idx = 0; idx < jec_.size(); ++idx ) {
    if ( set.empty() || jec_.at(idx).jecSet() == set ){
      if ( jec_[idx].jecLevel(level) >= 0 ) 
	return jecFactor(jec_[idx].jecLevel(level), idx);
      else
	throw cms::Exception("InvalidRequest") 
	  << "This JEC level " << level << " does not exist. \n";
    }
  }
  throw cms::Exception("InvalidRequest") 
    << "This jet does not carry any jet energy correction factor information \n"
    << "for a jet energy correction set with label " << set << "\n";
}

/// correction factor to the given level for a specific set
/// of correction factors, starting from the current level
float Tau::jecFactor(const unsigned int& level, const unsigned int& set) const
{
  if ( !jecSetsAvailable() )
    throw cms::Exception("InvalidRequest") 
      << "This jet does not carry any jet energy correction factor information \n";
  if ( !jecSetAvailable(set) )
    throw cms::Exception("InvalidRequest") 
      << "This jet does not carry any jet energy correction factor information \n"
      << "for a jet energy correction set with index " << set << "\n";
  return jec_.at(set).correction(level)/jec_.at(currentJECSet_).correction(currentJECLevel_);
}

/// copy of the jet with correction factor to target step for
/// the set of correction factors, which is currently in use
Tau Tau::correctedTauJet(const std::string& level, const std::string& set) const
{
  // rescale p4 of the jet; the update of current values is
  // done within the called jecFactor function
  for ( unsigned int idx = 0; idx < jec_.size(); ++idx ) {
    if ( set.empty() || jec_.at(idx).jecSet() == set ) {
      if ( jec_[idx].jecLevel(level) >= 0 ) 
	return correctedTauJet(jec_[idx].jecLevel(level), idx);
      else
	throw cms::Exception("InvalidRequest") 
	  << "This JEC level " << level << " does not exist. \n";
    }
  }
  throw cms::Exception("InvalidRequest") 
    << "This JEC set " << set << " does not exist. \n";
}

/// copy of the jet with correction factor to target step for
/// the set of correction factors, which is currently in use
Tau Tau::correctedTauJet(const unsigned int& level, const unsigned int& set) const
{
  Tau correctedTauJet(*this);
  //rescale p4 of the jet
  correctedTauJet.setP4(jecFactor(level, set)*p4());
  // update current level and set
  correctedTauJet.currentJECSet(set); 
  correctedTauJet.currentJECLevel(level); 
  return correctedTauJet;
}

