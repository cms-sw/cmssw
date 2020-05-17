//
// Original Author:  Fedor Ratnikov Feb. 16, 2007
//
// Correction which chains other corrections
//

#include "JetMETCorrections/Objects/interface/ChainedJetCorrector.h"

/// get correction using Jet information only
double ChainedJetCorrector::correction(const LorentzVector& fJet) const {
  LorentzVector jet = fJet;
  double result = 1;
  for (auto mCorrector : mCorrectors) {
    double scale = mCorrector->correction(jet);
    jet *= scale;
    result *= scale;
  }
  return result;
}

/// apply correction using Jet information only
double ChainedJetCorrector::correction(const reco::Jet& fJet) const {
  std::unique_ptr<reco::Jet> jet(dynamic_cast<reco::Jet*>(fJet.clone()));
  double result = 1;
  for (auto mCorrector : mCorrectors) {
    double scale = mCorrector->correction(*jet);
    jet->scaleEnergy(scale);
    result *= scale;
  }
  return result;
}

/// apply correction using all event information
double ChainedJetCorrector::correction(const reco::Jet& fJet,
                                       const edm::Event& fEvent,
                                       const edm::EventSetup& fSetup) const {
  std::unique_ptr<reco::Jet> jet(dynamic_cast<reco::Jet*>(fJet.clone()));
  double result = 1;
  for (auto mCorrector : mCorrectors) {
    double scale = mCorrector->correction(*jet, fEvent, fSetup);
    jet->scaleEnergy(scale);
    result *= scale;
  }
  return result;
}
/// apply correction using all event information and reference to the raw jet
double ChainedJetCorrector::correction(const reco::Jet& fJet,
                                       const edm::RefToBase<reco::Jet>& fJetRef,
                                       const edm::Event& fEvent,
                                       const edm::EventSetup& fSetup) const {
  std::unique_ptr<reco::Jet> jet(dynamic_cast<reco::Jet*>(fJet.clone()));
  double result = 1;
  for (auto mCorrector : mCorrectors) {
    double scale = mCorrector->correction(*jet, fJetRef, fEvent, fSetup);
    jet->scaleEnergy(scale);
    result *= scale;
  }
  return result;
}

/// if correction needs event information
bool ChainedJetCorrector::eventRequired() const {
  for (auto mCorrector : mCorrectors) {
    if (mCorrector->eventRequired())
      return true;
  }
  return false;
}

/// if correction needs jet reference
bool ChainedJetCorrector::refRequired() const {
  for (auto mCorrector : mCorrectors) {
    if (mCorrector->refRequired())
      return true;
  }
  return false;
}
