#include "RecoTauTag/RecoTau/interface/RecoTauConstructor.h"

namespace reco { namespace tau {

// Specializations
template<>
void RecoTauConstructor<reco::PFTau, reco::PFCandidate, reco::PFCandidate>::setJetRef(const JetBaseRef& jet) {
  tau_->setjetRef(jet.castTo<reco::PFJetRef>());
}

template<>
void RecoTauConstructor<reco::PFBaseTau, pat::PackedCandidate, reco::Candidate>::setJetRef(const JetBaseRef& jet) {
  tau_->setjetRef(jet);
}

} } // end reco::tau namespace
