//
//

#include "FWCore/Utilities/interface/EDMException.h"
#include "CommonTools/CandUtils/interface/pdgIdUtils.h"
#include "AnalysisDataFormats/TopObjects/interface/StGenEvent.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

StGenEvent::StGenEvent() {}

StGenEvent::StGenEvent(reco::GenParticleRefProd& parts, reco::GenParticleRefProd& inits) {
  parts_ = parts;
  initPartons_ = inits;
}

StGenEvent::~StGenEvent() {}

const reco::GenParticle* StGenEvent::decayB() const {
  const reco::GenParticle* cand = nullptr;
  if (singleLepton()) {
    const reco::GenParticleCollection& partsColl = *parts_;
    const reco::GenParticle& singleLep = *singleLepton();
    for (unsigned int i = 0; i < parts_->size(); ++i) {
      if (std::abs(partsColl[i].pdgId()) == TopDecayID::bID &&
          reco::flavour(singleLep) == -reco::flavour(partsColl[i])) {
        // ... but it should be the opposite!
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* StGenEvent::associatedB() const {
  const reco::GenParticle* cand = nullptr;
  if (singleLepton()) {
    const reco::GenParticleCollection& partsColl = *parts_;
    const reco::GenParticle& singleLep = *singleLepton();
    for (unsigned int i = 0; i < parts_->size(); ++i) {
      if (std::abs(partsColl[i].pdgId()) == TopDecayID::bID &&
          reco::flavour(singleLep) == reco::flavour(partsColl[i])) {
        // ... but it should be the opposite!
        cand = &partsColl[i];
      }
    }
  }
  return cand;
}

const reco::GenParticle* StGenEvent::singleLepton() const {
  const reco::GenParticle* cand = nullptr;
  const reco::GenParticleCollection& partsColl = *parts_;
  for (const auto& i : partsColl) {
    if (reco::isLepton(i) && i.mother() && std::abs(i.mother()->pdgId()) == TopDecayID::WID) {
      cand = &i;
    }
  }
  return cand;
}

const reco::GenParticle* StGenEvent::singleNeutrino() const {
  const reco::GenParticle* cand = nullptr;
  const reco::GenParticleCollection& partsColl = *parts_;
  for (const auto& i : partsColl) {
    if (reco::isNeutrino(i) && i.mother() && std::abs(i.mother()->pdgId()) == TopDecayID::WID) {
      cand = &i;
    }
  }
  return cand;
}

const reco::GenParticle* StGenEvent::singleW() const {
  const reco::GenParticle* cand = nullptr;
  if (singleLepton()) {
    const reco::GenParticleCollection& partsColl = *parts_;
    const reco::GenParticle& singleLep = *singleLepton();
    for (const auto& i : partsColl) {
      if (std::abs(i.pdgId()) == TopDecayID::WID && reco::flavour(singleLep) == -reco::flavour(i)) {
        // PDG Id:13=mu- 24=W+ (+24)->(-13) (-24)->(+13) opposite sign
        cand = &i;
      }
    }
  }
  return cand;
}

const reco::GenParticle* StGenEvent::singleTop() const {
  const reco::GenParticle* cand = nullptr;
  if (singleLepton()) {
    const reco::GenParticleCollection& partsColl = *parts_;
    const reco::GenParticle& singleLep = *singleLepton();
    for (const auto& i : partsColl) {
      if (std::abs(i.pdgId()) == TopDecayID::tID && reco::flavour(singleLep) != reco::flavour(i)) {
        cand = &i;
      }
    }
  }
  return cand;
}
