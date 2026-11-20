//
// Package:     DataFormats/PatCandidates
// Class:       pat::GenJet (pat::io_v1::GenJet)
//
// Out-of-line members. Bodies marked "copy" follow DataFormats/PatCandidates/src/Jet.cc
// (CMSSW_20_1_0_pre3).
//
#include "DataFormats/PatCandidates/interface/GenJet.h"

#include <iomanip>

using namespace pat;

// ------------------------------------------------------------------ constructors

namespace {
  /// Ref to the reco::GenJet behind a RefToBase/Ptr into a std::vector<reco::GenJet>; null if the input is null
  template <typename R>
  reco::GenJetRef toGenJetRef(R const& r) {
    return r.isNonnull() ? reco::GenJetRef(r.id(), r.get(), r.key()) : reco::GenJetRef();
  }
}  // namespace

/// default constructor
GenJet::GenJet() : PATObject<reco::GenJet>(reco::GenJet()), jetCharge_(0.) {}

/// constructor from a reco::GenJet: keeps p4, vertex, area, constituents and reco::GenJet::Specific
GenJet::GenJet(const reco::GenJet& aJet) : PATObject<reco::GenJet>(aJet), jetCharge_(0.) {}

/// constructor from ref to reco::GenJet; forward and back ref of genJetFwdRef() point to it
GenJet::GenJet(const edm::RefToBase<reco::GenJet>& aJetRef)
    : PATObject<reco::GenJet>(aJetRef), genJetFwdRef_(toGenJetRef(aJetRef), toGenJetRef(aJetRef)), jetCharge_(0.) {}

/// constructor from ptr to reco::GenJet; forward and back ref of genJetFwdRef() point to it
GenJet::GenJet(const edm::Ptr<reco::GenJet>& aJetRef)
    : PATObject<reco::GenJet>(aJetRef), genJetFwdRef_(toGenJetRef(aJetRef), toGenJetRef(aJetRef)), jetCharge_(0.) {}

/// constructor from ref to pat::GenJet (copy of Jet.cc, pat::Jet(RefToBase<pat::Jet>))
GenJet::GenJet(const edm::RefToBase<pat::GenJet>& aJetRef) : GenJet(*aJetRef) {
  refToOrig_ = edm::Ptr<reco::Candidate>(aJetRef.id(), aJetRef.get(), aJetRef.key());
}

/// constructor from ptr to pat::GenJet (copy of Jet.cc, pat::Jet(Ptr<pat::Jet>))
GenJet::GenJet(const edm::Ptr<pat::GenJet>& aJetRef) : GenJet(*aJetRef) { refToOrig_ = aJetRef; }

GenJet::~GenJet() {}

std::ostream& reco::operator<<(std::ostream& out, const pat::GenJet& obj) {
  if (!out)
    return out;

  out << "\tpat::GenJet: ";
  out << std::setiosflags(std::ios::right);
  out << std::setiosflags(std::ios::fixed);
  out << std::setprecision(3);
  out << " E/pT/eta/phi " << obj.energy() << "/" << obj.pt() << "/" << obj.eta() << "/" << obj.phi();
  return out;
}

// ------------------------------------------------------------------ JEC: a copy of the jet

GenJet GenJet::correctedJet(const std::string& /*level*/,
                            const std::string& /*flavor*/,
                            const std::string& /*set*/) const {
  return *this;
}

GenJet GenJet::correctedJet(const unsigned int& /*level*/,
                            const JetCorrFactors::Flavor& /*flavor*/,
                            const unsigned int& /*set*/) const {
  return *this;
}

// ------------------------------------------------------------------ generator particles (as the PF candidates of Jet.cc, no throw)

void GenJet::setGenCandidates(const reco::GenParticleFwdPtrVector& genCandidates) {
  genCandidatesFwdPtr_.reserve(genCandidatesFwdPtr_.size() + genCandidates.size());
  for (auto const& cand : genCandidates) {
    genCandidatesFwdPtr_.push_back(cand);
  }
  genCandidatesTemp_.reset();
}

std::vector<reco::GenParticlePtr> const& GenJet::genCandidates() const {
  if (!genCandidatesTemp_.isSet())
    cacheGenCandidates();
  return *genCandidatesTemp_;
}

void GenJet::cacheGenCandidates() const {
  auto genCandidatesTemp = std::make_unique<std::vector<reco::GenParticlePtr>>();
  genCandidatesTemp->reserve(genCandidatesFwdPtr_.size());
  for (auto const& cand : genCandidatesFwdPtr_)
    genCandidatesTemp->emplace_back(cand.ptr());
  genCandidatesTemp_.set(std::move(genCandidatesTemp));
}

void GenJet::updateFwdGenCandidateFwdPtr(unsigned int index, const reco::GenParticlePtr& updateFwd) {
  if (index < genCandidatesFwdPtr_.size()) {
    genCandidatesFwdPtr_[index] = reco::GenParticleFwdPtr(updateFwd, genCandidatesFwdPtr_[index].backPtr());
    genCandidatesTemp_.reset();
  }
}

// ------------------------------------------------------------------ daughters (copy, subjet-aware)

const reco::Candidate* GenJet::daughter(size_t i) const {
  if (!subjetCollections_.empty()) {
    if (!daughtersTemp_.isSet())
      cacheDaughters();
    return daughtersTemp_->at(i).get();
  }
  return reco::GenJet::daughter(i);
}

reco::CandidatePtr GenJet::daughterPtr(size_t i) const {
  if (!subjetCollections_.empty()) {
    if (!daughtersTemp_.isSet())
      cacheDaughters();
    return daughtersTemp_->at(i);
  }
  return reco::GenJet::daughterPtr(i);
}

const reco::CompositePtrCandidate::daughters& GenJet::daughterPtrVector() const {
  if (!subjetCollections_.empty()) {
    if (!daughtersTemp_.isSet())
      cacheDaughters();
    return *daughtersTemp_;
  }
  return reco::GenJet::daughterPtrVector();
}

size_t GenJet::numberOfDaughters() const {
  if (!subjetCollections_.empty()) {
    if (!daughtersTemp_.isSet())
      cacheDaughters();
    return daughtersTemp_->size();
  }
  return reco::GenJet::numberOfDaughters();
}

/// copy of Jet.cc:616: subjet daughters are replaced by their own constituents. A null daughter (a constituent
/// without packed partner, kept as PATGenJetSlimmer does with keepUnmappedDaughters) is kept as it is.
void GenJet::cacheDaughters() const {
  auto daughtersTemp = std::make_unique<std::vector<reco::CandidatePtr>>();
  const std::vector<reco::CandidatePtr>& jdaus = reco::GenJet::daughterPtrVector();
  for (const reco::CandidatePtr& dau : jdaus) {
    if (dau.isNull() || !dau.id().isValid()) {
      daughtersTemp->push_back(dau);
    } else if (dau->isJet()) {
      const reco::Jet* subjet = dynamic_cast<const reco::Jet*>(&*dau);
      if (subjet) {
        const std::vector<reco::CandidatePtr>& sjdaus = subjet->daughterPtrVector();
        daughtersTemp->insert(daughtersTemp->end(), sjdaus.begin(), sjdaus.end());
      }
    } else
      daughtersTemp->push_back(dau);
  }
  daughtersTemp_.set(std::move(daughtersTemp));
}

// ------------------------------------------------------------------ subjets

pat::GenJetPtrCollection const& GenJet::emptySubjets() {
  static const pat::GenJetPtrCollection empty;
  return empty;
}

pat::GenJetPtrCollection const& GenJet::subjets(unsigned int index) const {
  if (index < subjetCollections_.size())
    return subjetCollections_[index];
  return emptySubjets();
}

pat::GenJetPtrCollection const& GenJet::subjets(std::string const& label) const {
  auto found = std::find(subjetLabels_.begin(), subjetLabels_.end(), label);
  if (found != subjetLabels_.end())
    return subjetCollections_[std::distance(subjetLabels_.begin(), found)];
  return emptySubjets();
}

void GenJet::addSubjets(pat::GenJetPtrCollection const& pieces, std::string const& label) {
  subjetCollections_.push_back(pieces);
  subjetLabels_.push_back(label);
}
