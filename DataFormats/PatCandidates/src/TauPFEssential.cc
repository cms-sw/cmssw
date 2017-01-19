#include "DataFormats/PatCandidates/interface/TauPFEssential.h"

#include "DataFormats/JetReco/interface/Jet.h"

pat::tau::TauPFEssential::TauPFEssential(const reco::PFTau& tau) :
    p4Jet_(reco::Candidate::LorentzVector()),
    p4CorrJet_(reco::Candidate::LorentzVector()),
    decayMode_(tau.decayMode()),
    dxy_(0.),
    dxy_error_(1.e+3),
    hasSV_(false),
    ip3d_(0.),
    ip3d_error_(1.e+3),
    ecalEnergy_(0.),
    hcalEnergy_(0.),
    leadingTrackNormChi2_(1.e+3),
    phiAtEcalEntrance_(0.),
    etaAtEcalEntrance_(0.),
    ecalEnergyLeadChargedHadrCand_(0.),
    hcalEnergyLeadChargedHadrCand_(0.),
    etaAtEcalEntranceLeadChargedCand_(0.),
    ptLeadChargedCand_(0.),
    emFraction_(0.)
{
  if ( tau.jetRef().isAvailable() && tau.jetRef().isNonnull() ) { // CV: add protection to ease transition to new CMSSW 4_2_x RecoTauTags
    p4Jet_ = tau.jetRef()->p4();
  }
}
