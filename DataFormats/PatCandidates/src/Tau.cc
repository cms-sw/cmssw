//
//

#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/JetReco/interface/GenJet.h"


using namespace pat;


/// default constructor
Tau::Tau() :
    Lepton<reco::BaseTau>()
    ,embeddedIsolationTracks_(false)
    ,embeddedLeadTrack_(false)
    ,embeddedSignalTracks_(false)
    ,embeddedLeadPFCand_(false)
    ,embeddedLeadPFChargedHadrCand_(false)
    ,embeddedLeadPFNeutralCand_(false)
    ,embeddedSignalPFCands_(false)
    ,embeddedSignalPFChargedHadrCands_(false)
    ,embeddedSignalPFNeutralHadrCands_(false)
    ,embeddedSignalPFGammaCands_(false)
    ,embeddedIsolationPFCands_(false)
    ,embeddedIsolationPFChargedHadrCands_(false)
    ,embeddedIsolationPFNeutralHadrCands_(false)
    ,embeddedIsolationPFGammaCands_(false)
{
}

/// constructor from reco::BaseTau
Tau::Tau(const reco::BaseTau & aTau) :
    Lepton<reco::BaseTau>(aTau)
    ,embeddedIsolationTracks_(false)
    ,embeddedLeadTrack_(false)
    ,embeddedSignalTracks_(false)
    ,embeddedLeadPFCand_(false)
    ,embeddedLeadPFChargedHadrCand_(false)
    ,embeddedLeadPFNeutralCand_(false)
    ,embeddedSignalPFCands_(false)
    ,embeddedSignalPFChargedHadrCands_(false)
    ,embeddedSignalPFNeutralHadrCands_(false)
    ,embeddedSignalPFGammaCands_(false)
    ,embeddedIsolationPFCands_(false)
    ,embeddedIsolationPFChargedHadrCands_(false)
    ,embeddedIsolationPFNeutralHadrCands_(false)
    ,embeddedIsolationPFGammaCands_(false)
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
    ,embeddedLeadTrack_(false)
    ,embeddedSignalTracks_(false)
    ,embeddedLeadPFCand_(false)
    ,embeddedLeadPFChargedHadrCand_(false)
    ,embeddedLeadPFNeutralCand_(false)
    ,embeddedSignalPFCands_(false)
    ,embeddedSignalPFChargedHadrCands_(false)
    ,embeddedSignalPFNeutralHadrCands_(false)
    ,embeddedSignalPFGammaCands_(false)
    ,embeddedIsolationPFCands_(false)
    ,embeddedIsolationPFChargedHadrCands_(false)
    ,embeddedIsolationPFNeutralHadrCands_(false)
    ,embeddedIsolationPFGammaCands_(false)
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
    ,embeddedLeadTrack_(false)
    ,embeddedSignalTracks_(false)
    ,embeddedLeadPFCand_(false)
    ,embeddedLeadPFChargedHadrCand_(false)
    ,embeddedLeadPFNeutralCand_(false)
    ,embeddedSignalPFCands_(false)
    ,embeddedSignalPFChargedHadrCands_(false)
    ,embeddedSignalPFNeutralHadrCands_(false)
    ,embeddedSignalPFGammaCands_(false)
    ,embeddedIsolationPFCands_(false)
    ,embeddedIsolationPFChargedHadrCands_(false)
    ,embeddedIsolationPFNeutralHadrCands_(false)
    ,embeddedIsolationPFGammaCands_(false)
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
    if (!isolationTracksTransientRefVector_.isSet()) {
	std::unique_ptr<reco::TrackRefVector> trackRefVec{ new reco::TrackRefVector{}};
	trackRefVec->reserve(isolationTracks_.size());
        for (unsigned int i = 0; i < isolationTracks_.size(); i++) {
          trackRefVec->push_back(reco::TrackRef(&isolationTracks_, i));
        }
	isolationTracksTransientRefVector_.set(std::move(trackRefVec));
    }
    return *isolationTracksTransientRefVector_;
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
    if (!signalTracksTransientRefVector_.isSet()) {
        std::unique_ptr<reco::TrackRefVector> trackRefVec{ new reco::TrackRefVector{} };
        trackRefVec->reserve(signalTracks_.size());
        for (unsigned int i = 0; i < signalTracks_.size(); i++) {
          trackRefVec->push_back(reco::TrackRef(&signalTracks_, i));
        }
	signalTracksTransientRefVector_.set(std::move(trackRefVec));
    }
    return *signalTracksTransientRefVector_;
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

double Tau::dxy_Sig() const
{
  if ( pfSpecific().dxy_error_ != 0 ) return (pfSpecific().dxy_/pfSpecific().dxy_error_);
  else return 0.;
}

reco::PFTauTransverseImpactParameter::CovMatrix Tau::flightLengthCov() const
{
  reco::PFTauTransverseImpactParameter::CovMatrix cov;
  const reco::PFTauTransverseImpactParameter::CovMatrix& sv = secondaryVertexCov();
  const reco::PFTauTransverseImpactParameter::CovMatrix& pv = primaryVertexCov();
  for ( int i = 0; i < dimension; ++i ) {
    for ( int j = 0; j < dimension; ++j ) {
      cov(i,j) = sv(i,j) + pv(i,j);
    }
  }
  return cov;
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
  std::vector<reco::PFCandidatePtr> candPtrs = pfSpecific_[0].selectedSignalPFCands_;
  for (unsigned int i = 0; i < candPtrs.size(); i++) {
    signalPFCands_.push_back(*candPtrs.at(i));
  }
  embeddedSignalPFCands_ = true;
}
void Tau::embedSignalPFChargedHadrCands() {
  if (!isPFTau() ) {//additional check with warning in pat::tau producer
    return;
  }
  std::vector<reco::PFCandidatePtr> candPtrs = pfSpecific_[0].selectedSignalPFChargedHadrCands_;
  for (unsigned int i = 0; i < candPtrs.size(); i++) {
    signalPFChargedHadrCands_.push_back(*candPtrs.at(i));
  }
  embeddedSignalPFChargedHadrCands_ = true;
}
void Tau::embedSignalPFNeutralHadrCands() {
  if (!isPFTau() ) {//additional check with warning in pat::tau producer
    return;
  }
  std::vector<reco::PFCandidatePtr> candPtrs = pfSpecific_[0].selectedSignalPFNeutrHadrCands_;
  for (unsigned int i = 0; i < candPtrs.size(); i++) {
    signalPFNeutralHadrCands_.push_back(*candPtrs.at(i));
  }
  embeddedSignalPFNeutralHadrCands_ = true;
}
void Tau::embedSignalPFGammaCands() {
  if (!isPFTau() ) {//additional check with warning in pat::tau producer
    return;
  }
  std::vector<reco::PFCandidatePtr> candPtrs = pfSpecific_[0].selectedSignalPFGammaCands_;
  for (unsigned int i = 0; i < candPtrs.size(); i++) {
    signalPFGammaCands_.push_back(*candPtrs.at(i));
  }
  embeddedSignalPFGammaCands_ = true;
}

void Tau::embedIsolationPFCands() {
  if (!isPFTau() ) {//additional check with warning in pat::tau producer
    return;
  }
  std::vector<reco::PFCandidatePtr> candPtrs = pfSpecific_[0].selectedIsolationPFCands_;
  for (unsigned int i = 0; i < candPtrs.size(); i++) {
    isolationPFCands_.push_back(*candPtrs.at(i));
  }
  embeddedIsolationPFCands_ = true;
}

void Tau::embedIsolationPFChargedHadrCands() {
  if (!isPFTau() ) {//additional check with warning in pat::tau producer
    return;
  }
  std::vector<reco::PFCandidatePtr> candPtrs = pfSpecific_[0].selectedIsolationPFChargedHadrCands_;
  for (unsigned int i = 0; i < candPtrs.size(); i++) {
    isolationPFChargedHadrCands_.push_back(*candPtrs.at(i));
  }
  embeddedIsolationPFChargedHadrCands_ = true;
}
void Tau::embedIsolationPFNeutralHadrCands() {
  if (!isPFTau() ) {//additional check with warning in pat::tau producer
    return;
  }
  std::vector<reco::PFCandidatePtr> candPtrs = pfSpecific_[0].selectedIsolationPFNeutrHadrCands_;
  for (unsigned int i = 0; i < candPtrs.size(); i++) {
    isolationPFNeutralHadrCands_.push_back(*candPtrs.at(i));
  }
  embeddedIsolationPFNeutralHadrCands_ = true;
}
void Tau::embedIsolationPFGammaCands() {
  if (!isPFTau() ) {//additional check with warning in pat::tau producer
    return;
  }
  std::vector<reco::PFCandidatePtr> candPtrs = pfSpecific_[0].selectedIsolationPFGammaCands_;
  for (unsigned int i = 0; i < candPtrs.size(); i++) {
    isolationPFGammaCands_.push_back(*candPtrs.at(i));
  }
  embeddedIsolationPFGammaCands_ = true;
}

reco::PFRecoTauChargedHadronRef Tau::leadTauChargedHadronCandidate() const {
  if ( pfSpecific().signalTauChargedHadronCandidates_.size() > 0 ) {
    return reco::PFRecoTauChargedHadronRef(&pfSpecific().signalTauChargedHadronCandidates_,0);
  } else {
    return reco::PFRecoTauChargedHadronRef();
  }
}

const reco::PFCandidatePtr Tau::leadPFChargedHadrCand() const { 
  if(!embeddedLeadPFChargedHadrCand_)
    return pfSpecific().leadPFChargedHadrCand_; 
  else
    return reco::PFCandidatePtr(&leadPFChargedHadrCand_,0);
}

const reco::PFCandidatePtr Tau::leadPFNeutralCand() const { 
  if(!embeddedLeadPFNeutralCand_)
    return pfSpecific().leadPFNeutralCand_;
  else
    return reco::PFCandidatePtr(&leadPFNeutralCand_,0);
}

const reco::PFCandidatePtr Tau::leadPFCand() const { 
  if(!embeddedLeadPFCand_)
    return pfSpecific().leadPFCand_;
  else
    return reco::PFCandidatePtr(&leadPFCand_,0);
}

const std::vector<reco::PFCandidatePtr>& Tau::signalPFCands() const { 
  if (embeddedSignalPFCands_) {
   if (!signalPFCandsTransientPtrs_.isSet()) {
      std::unique_ptr<std::vector<reco::PFCandidatePtr> > aPtrs{new std::vector<reco::PFCandidatePtr>{}};
      aPtrs->reserve(signalPFCands_.size());
      for (unsigned int i = 0; i < signalPFCands_.size(); i++) {
	aPtrs->push_back(reco::PFCandidatePtr(&signalPFCands_, i) );
      }
      signalPFCandsTransientPtrs_.set(std::move(aPtrs));
    }
    return *signalPFCandsTransientPtrs_;
  } else
    return pfSpecific().selectedSignalPFCands_; 
}

const std::vector<reco::PFCandidatePtr>& Tau::signalPFChargedHadrCands() const {
  if (embeddedSignalPFChargedHadrCands_) {
   if (!signalPFChargedHadrCandsTransientPtrs_.isSet()) {
      std::unique_ptr<std::vector<reco::PFCandidatePtr> > aPtrs{ new std::vector<reco::PFCandidatePtr>{}};
      aPtrs->reserve(signalPFChargedHadrCands_.size());
      for (unsigned int i = 0; i < signalPFChargedHadrCands_.size(); i++) {
	aPtrs->push_back(reco::PFCandidatePtr(&signalPFChargedHadrCands_, i) );
      }
      signalPFNeutralHadrCandsTransientPtrs_.set(std::move(aPtrs));
    }
    return *signalPFChargedHadrCandsTransientPtrs_;
  } else
    return pfSpecific().selectedSignalPFChargedHadrCands_;
} 

const std::vector<reco::PFCandidatePtr>& Tau::signalPFNeutrHadrCands() const {
  if (embeddedSignalPFNeutralHadrCands_) {
   if (!signalPFNeutralHadrCandsTransientPtrs_.isSet()) {
      std::unique_ptr<std::vector<reco::PFCandidatePtr> > aPtrs{new std::vector<reco::PFCandidatePtr>{}};
      aPtrs->reserve(signalPFNeutralHadrCands_.size());
      for (unsigned int i = 0; i < signalPFNeutralHadrCands_.size(); i++) {
	aPtrs->push_back(reco::PFCandidatePtr(&signalPFNeutralHadrCands_, i) );
      }
      signalPFNeutralHadrCandsTransientPtrs_.set(std::move(aPtrs));
    }
    return *signalPFNeutralHadrCandsTransientPtrs_;
  } else
    return pfSpecific().selectedSignalPFNeutrHadrCands_;
} 

const std::vector<reco::PFCandidatePtr>& Tau::signalPFGammaCands() const {
  if (embeddedSignalPFGammaCands_) {
   if (!signalPFGammaCandsTransientPtrs_.isSet()) {
      std::unique_ptr<std::vector<reco::PFCandidatePtr> > aPtrs{ new std::vector<reco::PFCandidatePtr>{}};
      aPtrs->reserve(signalPFGammaCands_.size());
      for (unsigned int i = 0; i < signalPFGammaCands_.size(); i++) {
	aPtrs->push_back(reco::PFCandidatePtr(&signalPFGammaCands_, i) );
      }
      signalPFGammaCandsTransientPtrs_.set(std::move(aPtrs));
    }
    return *signalPFGammaCandsTransientPtrs_;
  } else
    return pfSpecific().selectedSignalPFGammaCands_;
}

const std::vector<reco::PFRecoTauChargedHadron> & Tau::signalTauChargedHadronCandidates() const {
  return pfSpecific().signalTauChargedHadronCandidates_;
}

const std::vector<reco::RecoTauPiZero> & Tau::signalPiZeroCandidates() const {
  return pfSpecific().signalPiZeroCandidates_;
}

const std::vector<reco::PFCandidatePtr>& Tau::isolationPFCands() const {
  if (embeddedIsolationPFCands_) {
  if (!isolationPFCandsTransientPtrs_.isSet()) {
      std::unique_ptr<std::vector<reco::PFCandidatePtr> > aPtrs{ new std::vector<reco::PFCandidatePtr>{}};
      aPtrs->reserve(isolationPFCands_.size());
      for (unsigned int i = 0; i < isolationPFCands_.size(); i++) {
	aPtrs->push_back(reco::PFCandidatePtr(&isolationPFCands_, i) );
      }
      isolationPFChargedHadrCandsTransientPtrs_.set(std::move(aPtrs));
    }
    return *isolationPFCandsTransientPtrs_;
  } else
    return pfSpecific().selectedIsolationPFCands_;
} 

const std::vector<reco::PFCandidatePtr>& Tau::isolationPFChargedHadrCands() const {
  if (embeddedIsolationPFChargedHadrCands_) {
   if (!isolationPFChargedHadrCandsTransientPtrs_.isSet()) {
      std::unique_ptr<std::vector<reco::PFCandidatePtr> > aPtrs{ new std::vector<reco::PFCandidatePtr>{}};
      aPtrs->reserve(isolationPFChargedHadrCands_.size());
      for (unsigned int i = 0; i < isolationPFChargedHadrCands_.size(); i++) {
	aPtrs->push_back(reco::PFCandidatePtr(&isolationPFChargedHadrCands_, i) );
      }
      isolationPFChargedHadrCandsTransientPtrs_.set(std::move(aPtrs));
    }
    return *isolationPFChargedHadrCandsTransientPtrs_;
  } else
    return pfSpecific().selectedIsolationPFChargedHadrCands_;
} 

const std::vector<reco::PFCandidatePtr>& Tau::isolationPFNeutrHadrCands() const {
  if (embeddedIsolationPFNeutralHadrCands_) {
    if (!isolationPFNeutralHadrCandsTransientPtrs_.isSet()) {
      std::unique_ptr<std::vector<reco::PFCandidatePtr> > aPtrs{ new std::vector<reco::PFCandidatePtr>{}};
      aPtrs->reserve(isolationPFNeutralHadrCands_.size());
      for (unsigned int i = 0; i < isolationPFNeutralHadrCands_.size(); i++) {
	aPtrs->push_back(reco::PFCandidatePtr(&isolationPFNeutralHadrCands_, i) );
      }
      isolationPFGammaCandsTransientPtrs_.set(std::move(aPtrs));
    }
    return *isolationPFNeutralHadrCandsTransientPtrs_;
  } else
    return pfSpecific().selectedIsolationPFNeutrHadrCands_;
} 

const std::vector<reco::PFCandidatePtr>& Tau::isolationPFGammaCands() const {
  if (embeddedIsolationPFGammaCands_) {
    if (!isolationPFGammaCandsTransientPtrs_.isSet()) {
      std::unique_ptr<std::vector<reco::PFCandidatePtr> > aPtrs{new std::vector<reco::PFCandidatePtr>{}};
      aPtrs->reserve(isolationPFGammaCands_.size());
      for (unsigned int i = 0; i < isolationPFGammaCands_.size(); i++) {
	aPtrs->push_back(reco::PFCandidatePtr(&isolationPFGammaCands_, i) );
      }
      isolationPFGammaCandsTransientPtrs_.set(std::move(aPtrs));
    }
    return *isolationPFGammaCandsTransientPtrs_;
  } else
    return pfSpecific().selectedIsolationPFGammaCands_;
}

const std::vector<reco::PFRecoTauChargedHadron> & Tau::isolationTauChargedHadronCandidates() const {
  return pfSpecific().isolationTauChargedHadronCandidates_;
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

