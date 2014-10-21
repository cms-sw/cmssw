#include "DataFormats/PatCandidates/interface/TauPFEssential.h"

#include "DataFormats/JetReco/interface/Jet.h"

pat::tau::TauPFEssential::TauPFEssential(const reco::PFTau& tau) :
    p4Jet_(reco::Candidate::LorentzVector()),
    p4CorrJet_(reco::Candidate::LorentzVector()),
    decayMode_(tau.decayMode()),
    dxy_(0.),
    dxy_error_(1.e+3),
    hasSV_(false)
{
  if ( tau.jetRef().isAvailable() && tau.jetRef().isNonnull() ) { // CV: add protection to ease transition to new CMSSW 4_2_x RecoTauTags
    p4Jet_ = tau.jetRef()->p4();
  }
}
