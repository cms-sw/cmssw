//
//

#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include <algorithm>
#include <functional>

using namespace pat;
using namespace std::placeholders;

/// default constructor
Tau::Tau()
    : Lepton<reco::BaseTau>(),
      embeddedIsolationTracks_(false),
      embeddedLeadTrack_(false),
      embeddedSignalTracks_(false),
      embeddedLeadPFCand_(false),
      embeddedLeadPFChargedHadrCand_(false),
      embeddedLeadPFNeutralCand_(false),
      embeddedSignalPFCands_(false),
      embeddedSignalPFChargedHadrCands_(false),
      embeddedSignalPFNeutralHadrCands_(false),
      embeddedSignalPFGammaCands_(false),
      embeddedIsolationPFCands_(false),
      embeddedIsolationPFChargedHadrCands_(false),
      embeddedIsolationPFNeutralHadrCands_(false),
      embeddedIsolationPFGammaCands_(false) {}

/// constructor from reco::BaseTau
Tau::Tau(const reco::BaseTau& aTau)
    : Lepton<reco::BaseTau>(aTau),
      embeddedIsolationTracks_(false),
      embeddedLeadTrack_(false),
      embeddedSignalTracks_(false),
      embeddedLeadPFCand_(false),
      embeddedLeadPFChargedHadrCand_(false),
      embeddedLeadPFNeutralCand_(false),
      embeddedSignalPFCands_(false),
      embeddedSignalPFChargedHadrCands_(false),
      embeddedSignalPFNeutralHadrCands_(false),
      embeddedSignalPFGammaCands_(false),
      embeddedIsolationPFCands_(false),
      embeddedIsolationPFChargedHadrCands_(false),
      embeddedIsolationPFNeutralHadrCands_(false),
      embeddedIsolationPFGammaCands_(false) {
  initFromBaseTau(aTau);
}

/// constructor from ref to reco::BaseTau
Tau::Tau(const edm::RefToBase<reco::BaseTau>& aTauRef)
    : Lepton<reco::BaseTau>(aTauRef),
      embeddedIsolationTracks_(false),
      embeddedLeadTrack_(false),
      embeddedSignalTracks_(false),
      embeddedLeadPFCand_(false),
      embeddedLeadPFChargedHadrCand_(false),
      embeddedLeadPFNeutralCand_(false),
      embeddedSignalPFCands_(false),
      embeddedSignalPFChargedHadrCands_(false),
      embeddedSignalPFNeutralHadrCands_(false),
      embeddedSignalPFGammaCands_(false),
      embeddedIsolationPFCands_(false),
      embeddedIsolationPFChargedHadrCands_(false),
      embeddedIsolationPFNeutralHadrCands_(false),
      embeddedIsolationPFGammaCands_(false) {
  initFromBaseTau(*aTauRef);
}

/// constructor from ref to reco::BaseTau
Tau::Tau(const edm::Ptr<reco::BaseTau>& aTauRef)
    : Lepton<reco::BaseTau>(aTauRef),
      embeddedIsolationTracks_(false),
      embeddedLeadTrack_(false),
      embeddedSignalTracks_(false),
      embeddedLeadPFCand_(false),
      embeddedLeadPFChargedHadrCand_(false),
      embeddedLeadPFNeutralCand_(false),
      embeddedSignalPFCands_(false),
      embeddedSignalPFChargedHadrCands_(false),
      embeddedSignalPFNeutralHadrCands_(false),
      embeddedSignalPFGammaCands_(false),
      embeddedIsolationPFCands_(false),
      embeddedIsolationPFChargedHadrCands_(false),
      embeddedIsolationPFNeutralHadrCands_(false),
      embeddedIsolationPFGammaCands_(false) {
  initFromBaseTau(*aTauRef);
}

void Tau::initFromBaseTau(const reco::BaseTau& aTau) {
  const reco::PFTau* pfTau = dynamic_cast<const reco::PFTau*>(&aTau);
  if (pfTau != nullptr) {
    // If PFTau is made from PackedCandidates, directly fill slimmed version
    // without PFSpecific
    const pat::PackedCandidate* pc = dynamic_cast<const pat::PackedCandidate*>(pfTau->leadChargedHadrCand().get());
    if (pc != nullptr) {
      for (const auto& ptr : pfTau->signalChargedHadrCands())
        signalChargedHadrCandPtrs_.push_back(ptr);

      for (const auto& ptr : pfTau->signalNeutrHadrCands())
        signalNeutralHadrCandPtrs_.push_back(ptr);

      for (const auto& ptr : pfTau->signalGammaCands())
        signalGammaCandPtrs_.push_back(ptr);

      for (const auto& ptr : pfTau->isolationChargedHadrCands())
        isolationChargedHadrCandPtrs_.push_back(ptr);

      for (const auto& ptr : pfTau->isolationNeutrHadrCands())
        isolationNeutralHadrCandPtrs_.push_back(ptr);

      for (const auto& ptr : pfTau->isolationGammaCands())
        isolationGammaCandPtrs_.push_back(ptr);

      std::vector<reco::CandidatePtr> signalLostTracks;
      for (const auto& chargedHadron : pfTau->signalTauChargedHadronCandidates()) {
        if (chargedHadron.algoIs(reco::PFRecoTauChargedHadron::kTrack) &&
            chargedHadron.getLostTrackCandidate().isNonnull()) {
          signalLostTracks.push_back(chargedHadron.getLostTrackCandidate());
        }
      }
      this->setSignalLostTracks(signalLostTracks);
    } else {
      pfSpecific_.push_back(pat::tau::TauPFSpecific(*pfTau));
    }
    pfEssential_.push_back(pat::tau::TauPFEssential(*pfTau));
  }
}

/// destructor
Tau::~Tau() {}

std::ostream& reco::operator<<(std::ostream& out, const pat::Tau& obj) {
  if (!out)
    return out;

  out << "\tpat::Tau: ";
  out << std::setiosflags(std::ios::right);
  out << std::setiosflags(std::ios::fixed);
  out << std::setprecision(3);
  out << " E/pT/eta/phi " << obj.energy() << "/" << obj.pt() << "/" << obj.eta() << "/" << obj.phi();
  return out;
}

/// override the reco::BaseTau::isolationTracks method, to access the internal storage of the track
const reco::TrackRefVector& Tau::isolationTracks() const {
  if (embeddedIsolationTracks_) {
    if (!isolationTracksTransientRefVector_.isSet()) {
      std::unique_ptr<reco::TrackRefVector> trackRefVec{new reco::TrackRefVector{}};
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
const reco::TrackRefVector& Tau::signalTracks() const {
  if (embeddedSignalTracks_) {
    if (!signalTracksTransientRefVector_.isSet()) {
      std::unique_ptr<reco::TrackRefVector> trackRefVec{new reco::TrackRefVector{}};
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
void Tau::embedSignalTracks() {
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
const reco::GenJet* Tau::genJet() const { return (!genJet_.empty() ? &genJet_.front() : nullptr); }

// method to retrieve a tau ID (or throw)
float Tau::tauID(const std::string& name) const {
  for (std::vector<IdPair>::const_iterator it = tauIDs_.begin(), ed = tauIDs_.end(); it != ed; ++it) {
    if (it->first == name)
      return it->second;
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
bool Tau::isTauIDAvailable(const std::string& name) const {
  for (std::vector<IdPair>::const_iterator it = tauIDs_.begin(), ed = tauIDs_.end(); it != ed; ++it) {
    if (it->first == name)
      return true;
  }
  return false;
}

const pat::tau::TauPFSpecific& Tau::pfSpecific() const {
  if (!isPFTau())
    throw cms::Exception("Type Error")
        << "Requesting a PFTau-specific information from a pat::Tau which wasn't made from a PFTau.\n";
  return pfSpecific_[0];
}

const pat::tau::TauPFEssential& Tau::pfEssential() const {
  if (pfEssential_.empty())
    throw cms::Exception("Type Error")
        << "Requesting a PFTau-specific information from a pat::Tau which wasn't made from a PFTau.\n";
  return pfEssential_[0];
}

reco::Candidate::LorentzVector Tau::p4Jet() const {
  if (isPFTau())
    return reco::Candidate::LorentzVector(pfEssential().p4Jet_);
  throw cms::Exception("Type Error") << "Requesting a PFTau-specific information from a pat::Tau which wasn't "
                                        "made from a PFTau.\n";
}

float Tau::dxy_Sig() const {
  if (pfEssential().dxy_error_ != 0)
    return (pfEssential().dxy_ / pfEssential().dxy_error_);
  else
    return 0.;
}

pat::tau::TauPFEssential::CovMatrix Tau::flightLengthCov() const {
  pat::tau::TauPFEssential::CovMatrix cov;
  const pat::tau::TauPFEssential::CovMatrix& sv = secondaryVertexCov();
  const pat::tau::TauPFEssential::CovMatrix& pv = primaryVertexCov();
  for (int i = 0; i < dimension; ++i) {
    for (int j = 0; j < dimension; ++j) {
      cov(i, j) = sv(i, j) + pv(i, j);
    }
  }
  return cov;
}

float Tau::ip3d_Sig() const {
  if (pfEssential().ip3d_error_ != 0)
    return (pfEssential().ip3d_ / pfEssential().ip3d_error_);
  else
    return 0.;
}

float Tau::etaetaMoment() const {
  if (isPFTau())
    return pfSpecific().etaetaMoment_;
  throw cms::Exception("Type Error") << "Requesting a PFTau-specific information from a pat::Tau which wasn't "
                                        "made from a PFTau.\n";
}

float Tau::phiphiMoment() const {
  if (isPFTau())
    return pfSpecific().phiphiMoment_;
  throw cms::Exception("Type Error") << "Requesting a PFTau-specific information from a pat::Tau which wasn't "
                                        "made from a PFTau.\n";
}

float Tau::etaphiMoment() const {
  if (isPFTau())
    return pfSpecific().etaphiMoment_;
  throw cms::Exception("Type Error") << "Requesting a PFTau-specific information from a pat::Tau which wasn't "
                                        "made from a PFTau.\n";
}

void Tau::setDecayMode(int decayMode) {
  if (!isPFTau())
    throw cms::Exception("Type Error")
        << "Requesting a PFTau-specific information from a pat::Tau which wasn't made from a PFTau.\n";
  pfEssential_[0].decayMode_ = decayMode;
}

/// method to store the leading candidate internally
void Tau::embedLeadPFCand() {
  if (!isPFTau()) {  //additional check with warning in pat::tau producer
    return;
  }
  leadPFCand_.clear();
  if (pfSpecific_[0].leadPFCand_.isNonnull()) {
    leadPFCand_.push_back(*static_cast<const reco::PFCandidate*>(&*pfSpecific_[0].leadPFCand_));  //already set in C-tor
    embeddedLeadPFCand_ = true;
  }
}
/// method to store the leading candidate internally
void Tau::embedLeadPFChargedHadrCand() {
  if (!isPFTau()) {  //additional check with warning in pat::tau producer
    return;
  }
  leadPFChargedHadrCand_.clear();
  if (pfSpecific_[0].leadPFChargedHadrCand_.isNonnull()) {
    leadPFChargedHadrCand_.push_back(
        *static_cast<const reco::PFCandidate*>(&*pfSpecific_[0].leadPFChargedHadrCand_));  //already set in C-tor
    embeddedLeadPFChargedHadrCand_ = true;
  }
}
/// method to store the leading candidate internally
void Tau::embedLeadPFNeutralCand() {
  if (!isPFTau()) {  //additional check with warning in pat::tau producer
    return;
  }
  leadPFNeutralCand_.clear();
  if (pfSpecific_[0].leadPFNeutralCand_.isNonnull()) {
    leadPFNeutralCand_.push_back(
        *static_cast<const reco::PFCandidate*>(&*pfSpecific_[0].leadPFNeutralCand_));  //already set in C-tor
    embeddedLeadPFNeutralCand_ = true;
  }
}

void Tau::embedSignalPFCands() {
  if (!isPFTau()) {  //additional check with warning in pat::tau producer
    return;
  }
  std::vector<reco::PFCandidatePtr> candPtrs = pfSpecific_[0].selectedSignalPFCands_;
  for (unsigned int i = 0; i < candPtrs.size(); i++) {
    signalPFCands_.push_back(candPtrs.at(i));
  }
  embeddedSignalPFCands_ = true;
}
void Tau::embedSignalPFChargedHadrCands() {
  if (!isPFTau()) {  //additional check with warning in pat::tau producer
    return;
  }
  std::vector<reco::PFCandidatePtr> candPtrs = pfSpecific_[0].selectedSignalPFChargedHadrCands_;
  for (unsigned int i = 0; i < candPtrs.size(); i++) {
    signalPFChargedHadrCands_.push_back(candPtrs.at(i));
  }
  embeddedSignalPFChargedHadrCands_ = true;
}
void Tau::embedSignalPFNeutralHadrCands() {
  if (!isPFTau()) {  //additional check with warning in pat::tau producer
    return;
  }
  std::vector<reco::PFCandidatePtr> candPtrs = pfSpecific_[0].selectedSignalPFNeutrHadrCands_;
  for (unsigned int i = 0; i < candPtrs.size(); i++) {
    signalPFNeutralHadrCands_.push_back(candPtrs.at(i));
  }
  embeddedSignalPFNeutralHadrCands_ = true;
}
void Tau::embedSignalPFGammaCands() {
  if (!isPFTau()) {  //additional check with warning in pat::tau producer
    return;
  }
  std::vector<reco::PFCandidatePtr> candPtrs = pfSpecific_[0].selectedSignalPFGammaCands_;
  for (unsigned int i = 0; i < candPtrs.size(); i++) {
    signalPFGammaCands_.push_back(candPtrs.at(i));
  }
  embeddedSignalPFGammaCands_ = true;
}

void Tau::embedIsolationPFCands() {
  if (!isPFTau()) {  //additional check with warning in pat::tau producer
    return;
  }
  std::vector<reco::PFCandidatePtr> candPtrs = pfSpecific_[0].selectedIsolationPFCands_;
  for (unsigned int i = 0; i < candPtrs.size(); i++) {
    isolationPFCands_.push_back(candPtrs.at(i));
  }
  embeddedIsolationPFCands_ = true;
}

void Tau::embedIsolationPFChargedHadrCands() {
  if (!isPFTau()) {  //additional check with warning in pat::tau producer
    return;
  }
  std::vector<reco::PFCandidatePtr> candPtrs = pfSpecific_[0].selectedIsolationPFChargedHadrCands_;
  for (unsigned int i = 0; i < candPtrs.size(); i++) {
    isolationPFChargedHadrCands_.push_back(candPtrs.at(i));
  }
  embeddedIsolationPFChargedHadrCands_ = true;
}
void Tau::embedIsolationPFNeutralHadrCands() {
  if (!isPFTau()) {  //additional check with warning in pat::tau producer
    return;
  }
  std::vector<reco::PFCandidatePtr> candPtrs = pfSpecific_[0].selectedIsolationPFNeutrHadrCands_;
  for (unsigned int i = 0; i < candPtrs.size(); i++) {
    isolationPFNeutralHadrCands_.push_back(candPtrs.at(i));
  }
  embeddedIsolationPFNeutralHadrCands_ = true;
}
void Tau::embedIsolationPFGammaCands() {
  if (!isPFTau()) {  //additional check with warning in pat::tau producer
    return;
  }
  std::vector<reco::PFCandidatePtr> candPtrs = pfSpecific_[0].selectedIsolationPFGammaCands_;
  for (unsigned int i = 0; i < candPtrs.size(); i++) {
    isolationPFGammaCands_.push_back(candPtrs.at(i));
  }
  embeddedIsolationPFGammaCands_ = true;
}

reco::PFRecoTauChargedHadronRef Tau::leadTauChargedHadronCandidate() const {
  if (!isPFTau())
    throw cms::Exception("Type Error") << "Requesting content that is not stored in miniAOD.\n";
  if (!pfSpecific().signalTauChargedHadronCandidates_.empty()) {
    return reco::PFRecoTauChargedHadronRef(&pfSpecific().signalTauChargedHadronCandidates_, 0);
  } else {
    return reco::PFRecoTauChargedHadronRef();
  }
}

const reco::PFCandidatePtr convertToPFCandidatePtr(const reco::CandidatePtr& ptr) {
  const reco::PFCandidate* pf_cand = dynamic_cast<const reco::PFCandidate*>(&*ptr);
  if (pf_cand)
    return edm::Ptr<reco::PFCandidate>(ptr);
  return reco::PFCandidatePtr();
}

const reco::PFCandidatePtr Tau::leadPFChargedHadrCand() const {
  if (!embeddedLeadPFChargedHadrCand_) {
    if (pfSpecific_.empty())
      return reco::PFCandidatePtr();
    else
      return convertToPFCandidatePtr(pfSpecific().leadPFChargedHadrCand_);
  } else
    return reco::PFCandidatePtr(&leadPFChargedHadrCand_, 0);
}

const reco::PFCandidatePtr Tau::leadPFNeutralCand() const {
  if (!embeddedLeadPFNeutralCand_) {
    if (pfSpecific_.empty())
      return reco::PFCandidatePtr();
    else
      return convertToPFCandidatePtr(pfSpecific().leadPFNeutralCand_);
  } else
    return reco::PFCandidatePtr(&leadPFNeutralCand_, 0);
}

const reco::PFCandidatePtr Tau::leadPFCand() const {
  if (!embeddedLeadPFCand_) {
    if (pfSpecific_.empty())
      return reco::PFCandidatePtr();
    return convertToPFCandidatePtr(pfSpecific().leadPFCand_);
  } else
    return reco::PFCandidatePtr(&leadPFCand_, 0);
}

const std::vector<reco::PFCandidatePtr>& Tau::signalPFCands() const {
  if (embeddedSignalPFCands_) {
    if (!signalPFCandsTransientPtrs_.isSet()) {
      std::unique_ptr<std::vector<reco::PFCandidatePtr> > aPtrs{new std::vector<reco::PFCandidatePtr>{}};
      aPtrs->reserve(signalPFCands_.size());
      for (unsigned int i = 0; i < signalPFCands_.size(); i++) {
        aPtrs->push_back(reco::PFCandidatePtr(&signalPFCands_, i));
      }
      signalPFCandsTransientPtrs_.set(std::move(aPtrs));
    }
    return *signalPFCandsTransientPtrs_;
  } else {
    if (pfSpecific_.empty() || pfSpecific().selectedSignalPFCands_.empty() ||
        !pfSpecific().selectedSignalPFCands_.front().isAvailable()) {
      // this part of code is called when reading from patTuple or miniAOD
      // it returns empty collection in correct format so it can be substituted by reco::Candidates if available
      if (!signalPFCandsTransientPtrs_.isSet()) {
        std::unique_ptr<std::vector<reco::PFCandidatePtr> > aPtrs{new std::vector<reco::PFCandidatePtr>{}};
        signalPFCandsTransientPtrs_.set(std::move(aPtrs));
      }
      return *signalPFCandsTransientPtrs_;
    } else
      return pfSpecific().selectedSignalPFCands_;
  }
}

const std::vector<reco::PFCandidatePtr>& Tau::signalPFChargedHadrCands() const {
  if (embeddedSignalPFChargedHadrCands_) {
    if (!signalPFChargedHadrCandsTransientPtrs_.isSet()) {
      std::unique_ptr<std::vector<reco::PFCandidatePtr> > aPtrs{new std::vector<reco::PFCandidatePtr>{}};
      aPtrs->reserve(signalPFChargedHadrCands_.size());
      for (unsigned int i = 0; i < signalPFChargedHadrCands_.size(); i++) {
        aPtrs->push_back(reco::PFCandidatePtr(&signalPFChargedHadrCands_, i));
      }
      signalPFChargedHadrCandsTransientPtrs_.set(std::move(aPtrs));
    }
    return *signalPFChargedHadrCandsTransientPtrs_;
  } else {
    if (pfSpecific_.empty() || pfSpecific().selectedSignalPFChargedHadrCands_.empty() ||
        !pfSpecific().selectedSignalPFChargedHadrCands_.front().isAvailable()) {
      // this part of code is called when reading from patTuple or miniAOD
      // it returns empty collection in correct format so it can be substituted by reco::Candidates if available
      if (!signalPFChargedHadrCandsTransientPtrs_.isSet()) {
        std::unique_ptr<std::vector<reco::PFCandidatePtr> > aPtrs{new std::vector<reco::PFCandidatePtr>{}};
        signalPFChargedHadrCandsTransientPtrs_.set(std::move(aPtrs));
      }
      return *signalPFChargedHadrCandsTransientPtrs_;
    } else
      return pfSpecific().selectedSignalPFChargedHadrCands_;
  }
}

const std::vector<reco::PFCandidatePtr>& Tau::signalPFNeutrHadrCands() const {
  if (embeddedSignalPFNeutralHadrCands_) {
    if (!signalPFNeutralHadrCandsTransientPtrs_.isSet()) {
      std::unique_ptr<std::vector<reco::PFCandidatePtr> > aPtrs{new std::vector<reco::PFCandidatePtr>{}};
      aPtrs->reserve(signalPFNeutralHadrCands_.size());
      for (unsigned int i = 0; i < signalPFNeutralHadrCands_.size(); i++) {
        aPtrs->push_back(reco::PFCandidatePtr(&signalPFNeutralHadrCands_, i));
      }
      signalPFNeutralHadrCandsTransientPtrs_.set(std::move(aPtrs));
    }
    return *signalPFNeutralHadrCandsTransientPtrs_;
  } else {
    if (pfSpecific_.empty() || pfSpecific().selectedSignalPFNeutrHadrCands_.empty() ||
        !pfSpecific().selectedSignalPFNeutrHadrCands_.front().isAvailable()) {
      // this part of code is called when reading from patTuple or miniAOD
      // it returns empty collection in correct format so it can be substituted by reco::Candidates if available
      if (!signalPFNeutralHadrCandsTransientPtrs_.isSet()) {
        std::unique_ptr<std::vector<reco::PFCandidatePtr> > aPtrs{new std::vector<reco::PFCandidatePtr>{}};
        signalPFNeutralHadrCandsTransientPtrs_.set(std::move(aPtrs));
      }
      return *signalPFNeutralHadrCandsTransientPtrs_;
    } else
      return pfSpecific().selectedSignalPFNeutrHadrCands_;
  }
}

const std::vector<reco::PFCandidatePtr>& Tau::signalPFGammaCands() const {
  if (embeddedSignalPFGammaCands_) {
    if (!signalPFGammaCandsTransientPtrs_.isSet()) {
      std::unique_ptr<std::vector<reco::PFCandidatePtr> > aPtrs{new std::vector<reco::PFCandidatePtr>{}};
      aPtrs->reserve(signalPFGammaCands_.size());
      for (unsigned int i = 0; i < signalPFGammaCands_.size(); i++) {
        aPtrs->push_back(reco::PFCandidatePtr(&signalPFGammaCands_, i));
      }
      signalPFGammaCandsTransientPtrs_.set(std::move(aPtrs));
    }
    return *signalPFGammaCandsTransientPtrs_;
  } else {
    if (pfSpecific_.empty() || pfSpecific().selectedSignalPFGammaCands_.empty() ||
        !pfSpecific().selectedSignalPFGammaCands_.front().isAvailable()) {
      // this part of code is called when reading from patTuple or miniAOD
      // it returns empty collection in correct format so it can be substituted by reco::Candidates if available
      if (!signalPFGammaCandsTransientPtrs_.isSet()) {
        std::unique_ptr<std::vector<reco::PFCandidatePtr> > aPtrs{new std::vector<reco::PFCandidatePtr>{}};
        signalPFGammaCandsTransientPtrs_.set(std::move(aPtrs));
      }
      return *signalPFGammaCandsTransientPtrs_;
    } else
      return pfSpecific().selectedSignalPFGammaCands_;
  }
}

const std::vector<reco::PFRecoTauChargedHadron>& Tau::signalTauChargedHadronCandidates() const {
  if (pfSpecific_.empty())
    throw cms::Exception("Type Error") << "Requesting content that is not stored in miniAOD.\n";
  return pfSpecific().signalTauChargedHadronCandidates_;
}

const std::vector<reco::RecoTauPiZero>& Tau::signalPiZeroCandidates() const {
  if (pfSpecific_.empty())
    throw cms::Exception("Type Error") << "Requesting content that is not stored in miniAOD.\n";
  return pfSpecific().signalPiZeroCandidates_;
}

const std::vector<reco::PFCandidatePtr>& Tau::isolationPFCands() const {
  if (embeddedIsolationPFCands_) {
    if (!isolationPFCandsTransientPtrs_.isSet()) {
      std::unique_ptr<std::vector<reco::PFCandidatePtr> > aPtrs{new std::vector<reco::PFCandidatePtr>{}};
      aPtrs->reserve(isolationPFCands_.size());
      for (unsigned int i = 0; i < isolationPFCands_.size(); i++) {
        aPtrs->push_back(reco::PFCandidatePtr(&isolationPFCands_, i));
      }
      isolationPFCandsTransientPtrs_.set(std::move(aPtrs));
    }
    return *isolationPFCandsTransientPtrs_;
  } else {
    if (pfSpecific_.empty() || pfSpecific().selectedIsolationPFCands_.empty() ||
        !pfSpecific().selectedIsolationPFCands_.front().isAvailable()) {
      // this part of code is called when reading from patTuple or miniAOD
      // it returns empty collection in correct format so it can be substituted by reco::Candidates if available
      if (!isolationPFCandsTransientPtrs_.isSet()) {
        std::unique_ptr<std::vector<reco::PFCandidatePtr> > aPtrs{new std::vector<reco::PFCandidatePtr>{}};
        isolationPFCandsTransientPtrs_.set(std::move(aPtrs));
      }
      return *isolationPFCandsTransientPtrs_;
    } else
      return pfSpecific().selectedIsolationPFCands_;
  }
}

const std::vector<reco::PFCandidatePtr>& Tau::isolationPFChargedHadrCands() const {
  if (embeddedIsolationPFChargedHadrCands_) {
    if (!isolationPFChargedHadrCandsTransientPtrs_.isSet()) {
      std::unique_ptr<std::vector<reco::PFCandidatePtr> > aPtrs{new std::vector<reco::PFCandidatePtr>{}};
      aPtrs->reserve(isolationPFChargedHadrCands_.size());
      for (unsigned int i = 0; i < isolationPFChargedHadrCands_.size(); i++) {
        aPtrs->push_back(reco::PFCandidatePtr(&isolationPFChargedHadrCands_, i));
      }
      isolationPFChargedHadrCandsTransientPtrs_.set(std::move(aPtrs));
    }
    return *isolationPFChargedHadrCandsTransientPtrs_;
  } else {
    if (pfSpecific_.empty() || pfSpecific().selectedIsolationPFChargedHadrCands_.empty() ||
        !pfSpecific().selectedIsolationPFChargedHadrCands_.front().isAvailable()) {
      // this part of code is called when reading from patTuple or miniAOD
      // it returns empty collection in correct format so it can be substituted by reco::Candidates if available
      if (!isolationPFChargedHadrCandsTransientPtrs_.isSet()) {
        std::unique_ptr<std::vector<reco::PFCandidatePtr> > aPtrs{new std::vector<reco::PFCandidatePtr>{}};
        isolationPFChargedHadrCandsTransientPtrs_.set(std::move(aPtrs));
      }
      return *isolationPFChargedHadrCandsTransientPtrs_;
    } else
      return pfSpecific().selectedIsolationPFChargedHadrCands_;
  }
}

const std::vector<reco::PFCandidatePtr>& Tau::isolationPFNeutrHadrCands() const {
  if (embeddedIsolationPFNeutralHadrCands_) {
    if (!isolationPFNeutralHadrCandsTransientPtrs_.isSet()) {
      std::unique_ptr<std::vector<reco::PFCandidatePtr> > aPtrs{new std::vector<reco::PFCandidatePtr>{}};
      aPtrs->reserve(isolationPFNeutralHadrCands_.size());
      for (unsigned int i = 0; i < isolationPFNeutralHadrCands_.size(); i++) {
        aPtrs->push_back(reco::PFCandidatePtr(&isolationPFNeutralHadrCands_, i));
      }
      isolationPFNeutralHadrCandsTransientPtrs_.set(std::move(aPtrs));
    }
    return *isolationPFNeutralHadrCandsTransientPtrs_;
  } else {
    if (pfSpecific_.empty() || pfSpecific().selectedIsolationPFNeutrHadrCands_.empty() ||
        !pfSpecific().selectedIsolationPFNeutrHadrCands_.front().isAvailable()) {
      // this part of code is called when reading from patTuple or miniAOD
      // it returns empty collection in correct format so it can be substituted by reco::Candidates if available
      if (!isolationPFNeutralHadrCandsTransientPtrs_.isSet()) {
        std::unique_ptr<std::vector<reco::PFCandidatePtr> > aPtrs{new std::vector<reco::PFCandidatePtr>{}};
        isolationPFNeutralHadrCandsTransientPtrs_.set(std::move(aPtrs));
      }
      return *isolationPFNeutralHadrCandsTransientPtrs_;
    } else
      return pfSpecific().selectedIsolationPFNeutrHadrCands_;
  }
}

const std::vector<reco::PFCandidatePtr>& Tau::isolationPFGammaCands() const {
  if (embeddedIsolationPFGammaCands_) {
    if (!isolationPFGammaCandsTransientPtrs_.isSet()) {
      std::unique_ptr<std::vector<reco::PFCandidatePtr> > aPtrs{new std::vector<reco::PFCandidatePtr>{}};
      aPtrs->reserve(isolationPFGammaCands_.size());
      for (unsigned int i = 0; i < isolationPFGammaCands_.size(); i++) {
        aPtrs->push_back(reco::PFCandidatePtr(&isolationPFGammaCands_, i));
      }
      isolationPFGammaCandsTransientPtrs_.set(std::move(aPtrs));
    }
    return *isolationPFGammaCandsTransientPtrs_;
  } else {
    if (pfSpecific_.empty() || pfSpecific().selectedIsolationPFGammaCands_.empty() ||
        !pfSpecific().selectedIsolationPFGammaCands_.front().isAvailable()) {
      // this part of code is called when reading from patTuple or miniAOD
      // it returns empty collection in correct format so it can be substituted by reco::Candidates if available
      if (!isolationPFGammaCandsTransientPtrs_.isSet()) {
        std::unique_ptr<std::vector<reco::PFCandidatePtr> > aPtrs{new std::vector<reco::PFCandidatePtr>{}};
        isolationPFGammaCandsTransientPtrs_.set(std::move(aPtrs));
      }
      return *isolationPFGammaCandsTransientPtrs_;
    } else
      return pfSpecific().selectedIsolationPFGammaCands_;
  }
}

const std::vector<reco::PFRecoTauChargedHadron>& Tau::isolationTauChargedHadronCandidates() const {
  if (pfSpecific_.empty())
    throw cms::Exception("Type Error") << "Requesting content that is not stored in miniAOD.\n";
  return pfSpecific().isolationTauChargedHadronCandidates_;
}

const std::vector<reco::RecoTauPiZero>& Tau::isolationPiZeroCandidates() const {
  if (pfSpecific_.empty())
    throw cms::Exception("Type Error") << "Requesting content that is not stored in miniAOD.\n";
  return pfSpecific().isolationPiZeroCandidates_;
}

/// ============= -Tau-jet Energy Correction methods ============
/// (copied from DataFormats/PatCandidates/src/Jet.cc)

// initialize the jet to a given JEC level during creation starting from Uncorrected
void Tau::initializeJEC(unsigned int level, unsigned int set) {
  currentJECSet(set);
  currentJECLevel(level);
  setP4(jec_[set].correction(level) * p4());
}

/// return true if this jet carries the jet correction factors of a different set, for systematic studies
int Tau::jecSet(const std::string& set) const {
  for (std::vector<pat::TauJetCorrFactors>::const_iterator corrFactor = jec_.begin(); corrFactor != jec_.end();
       ++corrFactor) {
    if (corrFactor->jecSet() == set)
      return corrFactor - jec_.begin();
  }
  return -1;
}

/// all available label-names of all sets of jet energy corrections
const std::vector<std::string> Tau::availableJECSets() const {
  std::vector<std::string> sets;
  for (std::vector<pat::TauJetCorrFactors>::const_iterator corrFactor = jec_.begin(); corrFactor != jec_.end();
       ++corrFactor) {
    sets.push_back(corrFactor->jecSet());
  }
  return sets;
}

const std::vector<std::string> Tau::availableJECLevels(const int& set) const {
  return set >= 0 ? jec_.at(set).correctionLabels() : std::vector<std::string>();
}

/// correction factor to the given level for a specific set
/// of correction factors, starting from the current level
float Tau::jecFactor(const std::string& level, const std::string& set) const {
  for (unsigned int idx = 0; idx < jec_.size(); ++idx) {
    if (set.empty() || jec_.at(idx).jecSet() == set) {
      if (jec_[idx].jecLevel(level) >= 0)
        return jecFactor(jec_[idx].jecLevel(level), idx);
      else
        throw cms::Exception("InvalidRequest") << "This JEC level " << level << " does not exist. \n";
    }
  }
  throw cms::Exception("InvalidRequest") << "This jet does not carry any jet energy correction factor information \n"
                                         << "for a jet energy correction set with label " << set << "\n";
}

/// correction factor to the given level for a specific set
/// of correction factors, starting from the current level
float Tau::jecFactor(const unsigned int& level, const unsigned int& set) const {
  if (!jecSetsAvailable())
    throw cms::Exception("InvalidRequest") << "This jet does not carry any jet energy correction factor information \n";
  if (!jecSetAvailable(set))
    throw cms::Exception("InvalidRequest") << "This jet does not carry any jet energy correction factor information \n"
                                           << "for a jet energy correction set with index " << set << "\n";
  return jec_.at(set).correction(level) / jec_.at(currentJECSet_).correction(currentJECLevel_);
}

/// copy of the jet with correction factor to target step for
/// the set of correction factors, which is currently in use
Tau Tau::correctedTauJet(const std::string& level, const std::string& set) const {
  // rescale p4 of the jet; the update of current values is
  // done within the called jecFactor function
  for (unsigned int idx = 0; idx < jec_.size(); ++idx) {
    if (set.empty() || jec_.at(idx).jecSet() == set) {
      if (jec_[idx].jecLevel(level) >= 0)
        return correctedTauJet(jec_[idx].jecLevel(level), idx);
      else
        throw cms::Exception("InvalidRequest") << "This JEC level " << level << " does not exist. \n";
    }
  }
  throw cms::Exception("InvalidRequest") << "This JEC set " << set << " does not exist. \n";
}

/// copy of the jet with correction factor to target step for
/// the set of correction factors, which is currently in use
Tau Tau::correctedTauJet(const unsigned int& level, const unsigned int& set) const {
  Tau correctedTauJet(*this);
  //rescale p4 of the jet
  correctedTauJet.setP4(jecFactor(level, set) * p4());
  // update current level and set
  correctedTauJet.currentJECSet(set);
  correctedTauJet.currentJECLevel(level);
  return correctedTauJet;
}

/// ----- Methods returning associated PFCandidates that work on PAT+AOD, PAT+embedding and miniAOD -----
/// return the PFCandidate if available (reference or embedded), or the PackedPFCandidate on miniAOD
// return the leading candidate from signal(PF)ChargedHadrCandPtrs_ collection
const reco::CandidatePtr Tau::leadChargedHadrCand() const {
  const reco::PFCandidatePtr leadPF = leadPFChargedHadrCand();
  if (leadPF.isAvailable() || signalChargedHadrCandPtrs_.isNull())
    return leadPF;
  reco::CandidatePtr ret;
  for (const reco::CandidatePtr& p : signalChargedHadrCandPtrs_) {
    if (ret.isNull() || (p->pt() > ret->pt()))
      ret = p;
  }
  return ret;
}

/// return the PFCandidate if available (reference or embedded), or the PackedPFCandidate on miniAOD
const reco::CandidatePtr Tau::leadNeutralCand() const {
  const reco::PFCandidatePtr leadPF = leadPFNeutralCand();
  if (leadPF.isAvailable() || signalNeutralHadrCandPtrs_.isNull())
    return leadPF;
  reco::CandidatePtr ret;
  for (const reco::CandidatePtr& p : signalNeutralHadrCandPtrs_) {
    if (ret.isNull() || (p->pt() > ret->pt()))
      ret = p;
  }
  return ret;
}

/// return the PFCandidate if available (reference or embedded), or the PackedPFCandidate on miniAOD
const reco::CandidatePtr Tau::leadCand() const {
  const reco::PFCandidatePtr leadPF = leadPFCand();
  if (leadPF.isAvailable() || !Tau::ExistSignalCands())
    return leadPF;
  else
    return Tau::signalCands()[0];
}

/// check that there is at least one non-zero collection of candidate ptrs

bool Tau::ExistSignalCands() const {
  return !(signalChargedHadrCandPtrs_.isNull() && signalNeutralHadrCandPtrs_.isNull() && signalGammaCandPtrs_.isNull());
}

bool Tau::ExistIsolationCands() const {
  return !(isolationChargedHadrCandPtrs_.isNull() && isolationNeutralHadrCandPtrs_.isNull() &&
           isolationGammaCandPtrs_.isNull());
}

/// return the PFCandidates if available (reference or embedded), or the PackedPFCandidate on miniAOD
/// note that the vector is returned by value.

reco::CandidatePtrVector Tau::signalCands() const {
  std::vector<reco::PFCandidatePtr> r0 = signalPFCands();
  reco::CandidatePtrVector ret;
  if (!Tau::ExistSignalCands() || (!r0.empty() && r0.front().isAvailable())) {
    for (const auto& p : r0)
      ret.push_back(p);
    return ret;
  } else {
    /// the isolationCands pointers are not saved in miniAOD, so the collection is created dynamically by glueing together 3 sub-collection and re-ordering
    reco::CandidatePtrVector ret2;
    std::vector<std::pair<float, size_t> > pt_index;
    size_t index = 0;
    ret2.reserve(signalChargedHadrCandPtrs_.size() + signalNeutralHadrCandPtrs_.size() + signalGammaCandPtrs_.size());
    pt_index.reserve(signalChargedHadrCandPtrs_.size() + signalNeutralHadrCandPtrs_.size() +
                     signalGammaCandPtrs_.size());

    for (const auto& p : signalChargedHadrCandPtrs_) {
      ret2.push_back(p);
      pt_index.push_back(std::make_pair(p->pt(), index));
      index++;
    }
    for (const auto& p : signalNeutralHadrCandPtrs_) {
      ret2.push_back(p);
      pt_index.push_back(std::make_pair(p->pt(), index));
      index++;
    }
    for (const auto& p : signalGammaCandPtrs_) {
      ret2.push_back(p);
      pt_index.push_back(std::make_pair(p->pt(), index));
      index++;
    }
    std::sort(pt_index.begin(), pt_index.end(), std::greater<>());
    ret.reserve(pt_index.size());
    for (const auto& p : pt_index) {
      ret.push_back(ret2[p.second]);
    }
    return ret;
  }
}

/// return the PFCandidates if available (reference or embedded), or the PackedPFCandidate on miniAOD
/// note that the vector is returned by value.
reco::CandidatePtrVector Tau::signalChargedHadrCands() const {
  std::vector<reco::PFCandidatePtr> r0 = signalPFChargedHadrCands();
  if (signalChargedHadrCandPtrs_.isNull() || (!r0.empty() && r0.front().isAvailable())) {
    reco::CandidatePtrVector ret;
    for (const auto& p : r0)
      ret.push_back(p);
    return ret;
  } else {
    return signalChargedHadrCandPtrs_;
  }
}

/// return the PFCandidates if available (reference or embedded), or the PackedPFCandidate on miniAOD
/// note that the vector is returned by value.
reco::CandidatePtrVector Tau::signalNeutrHadrCands() const {
  std::vector<reco::PFCandidatePtr> r0 = signalPFNeutrHadrCands();
  if (signalNeutralHadrCandPtrs_.isNull() || (!r0.empty() && r0.front().isAvailable())) {
    reco::CandidatePtrVector ret;
    for (const auto& p : r0)
      ret.push_back(p);
    return ret;
  } else {
    return signalNeutralHadrCandPtrs_;
  }
}

/// return the PFCandidates if available (reference or embedded), or the PackedPFCandidate on miniAOD
/// note that the vector is returned by value.
reco::CandidatePtrVector Tau::signalGammaCands() const {
  std::vector<reco::PFCandidatePtr> r0 = signalPFGammaCands();
  if (signalGammaCandPtrs_.isNull() || (!r0.empty() && r0.front().isAvailable())) {
    reco::CandidatePtrVector ret;
    for (const auto& p : r0)
      ret.push_back(p);
    return ret;
  } else {
    return signalGammaCandPtrs_;
  }
}

/// return the PFCandidates if available (reference or embedded), or the PackedPFCandidate on miniAOD
/// note that the vector is returned by value.
reco::CandidatePtrVector Tau::isolationCands() const {
  std::vector<reco::PFCandidatePtr> r0 = isolationPFCands();
  reco::CandidatePtrVector ret;
  if (!Tau::ExistIsolationCands() || (!r0.empty() && r0.front().isAvailable())) {
    for (const auto& p : r0)
      ret.push_back(p);
    return ret;
  } else {
    /// the isolationCands pointers are not saved in miniAOD, so the collection is created dynamically by glueing together 3 sub-collection and re-ordering
    reco::CandidatePtrVector ret2;
    std::vector<std::pair<float, size_t> > pt_index;
    ret2.reserve(isolationChargedHadrCandPtrs_.size() + isolationNeutralHadrCandPtrs_.size() +
                 isolationGammaCandPtrs_.size());
    pt_index.reserve(isolationChargedHadrCandPtrs_.size() + isolationNeutralHadrCandPtrs_.size() +
                     isolationGammaCandPtrs_.size());
    size_t index = 0;
    for (const auto& p : isolationChargedHadrCandPtrs_) {
      ret2.push_back(p);
      pt_index.push_back(std::make_pair(p->pt(), index));
      index++;
    }
    for (const auto& p : isolationNeutralHadrCandPtrs_) {
      ret2.push_back(p);
      pt_index.push_back(std::make_pair(p->pt(), index));
      index++;
    }
    for (const auto& p : isolationGammaCandPtrs_) {
      ret2.push_back(p);
      pt_index.push_back(std::make_pair(p->pt(), index));
      index++;
    }
    std::sort(pt_index.begin(), pt_index.end(), std::greater<>());
    ret.reserve(pt_index.size());
    for (const auto& p : pt_index) {
      ret.push_back(ret2[p.second]);
    }
    return ret;
  }
}

/// return the PFCandidates if available (reference or embedded), or the PackedPFCandidate on miniAOD
/// note that the vector is returned by value.
reco::CandidatePtrVector Tau::isolationChargedHadrCands() const {
  std::vector<reco::PFCandidatePtr> r0 = isolationPFChargedHadrCands();
  if (isolationChargedHadrCandPtrs_.isNull() || (!r0.empty() && r0.front().isAvailable())) {
    reco::CandidatePtrVector ret;
    for (const auto& p : r0)
      ret.push_back(p);
    return ret;
  } else {
    return isolationChargedHadrCandPtrs_;
  }
}

/// return the PFCandidates if available (reference or embedded), or the PackedPFCandidate on miniAOD
/// note that the vector is returned by value.
reco::CandidatePtrVector Tau::isolationNeutrHadrCands() const {
  reco::CandidatePtrVector ret;
  std::vector<reco::PFCandidatePtr> r0 = isolationPFNeutrHadrCands();
  if (isolationNeutralHadrCandPtrs_.isNull() || (!r0.empty() && r0.front().isAvailable())) {
    reco::CandidatePtrVector ret;
    for (const auto& p : r0)
      ret.push_back(p);
    return ret;
  } else {
    return isolationNeutralHadrCandPtrs_;
  }
}

/// return the PFCandidates if available (reference or embedded), or the PackedPFCandidate on miniAOD
/// note that the vector is returned by value.
reco::CandidatePtrVector Tau::isolationGammaCands() const {
  std::vector<reco::PFCandidatePtr> r0 = isolationPFGammaCands();
  if (isolationGammaCandPtrs_.isNull() || (!r0.empty() && r0.front().isAvailable())) {
    reco::CandidatePtrVector ret;
    for (const auto& p : r0)
      ret.push_back(p);
    return ret;
  } else {
    return isolationGammaCandPtrs_;
  }
}

std::vector<reco::CandidatePtr> Tau::signalLostTracks() const {
  std::vector<reco::CandidatePtr> ret;
  unsigned int i = 0;
  std::string label = "_lostTrack_" + std::to_string(i);
  while (this->hasUserCand(label)) {
    ret.push_back(userCand(label));
    i++;
    label = "_lostTrack_" + std::to_string(i);
  }
  return ret;
}

void Tau::setSignalLostTracks(const std::vector<reco::CandidatePtr>& ptrs) {
  unsigned int i = 0;
  for (const auto& ptr : ptrs) {
    std::string label = "_lostTrack_" + std::to_string(i);
    addUserCand(label, ptr);
    i++;
  }
}

/// ----- Top Projection business -------
/// get the number of non-null PFCandidates
size_t Tau::numberOfSourceCandidatePtrs() const {
  if (Tau::ExistSignalCands())
    return Tau::signalCands().size();
  else if (pfSpecific_.empty())
    return 0;
  else
    return pfSpecific().selectedSignalPFCands_.size();
}
/// get the source candidate pointer with index i
reco::CandidatePtr Tau::sourceCandidatePtr(size_type i) const {
  if (Tau::ExistSignalCands())
    return Tau::signalCands()[i];
  else if (pfSpecific_.empty())
    return reco::CandidatePtr();
  else
    return pfSpecific().selectedSignalPFCands_[i];
}
