//
// Original Author:  Fedor Ratnikov Feb. 16, 2007
// $Id: ChainedJetCorrector.cc,v 1.3 2011/04/28 14:05:22 kkousour Exp $
//
// Correction which chains other corrections
//

#include "JetMETCorrections/Objects/interface/ChainedJetCorrector.h"

/// get correction using Jet information only
double ChainedJetCorrector::correction (const LorentzVector& fJet) const
{
  LorentzVector jet = fJet;
  double result = 1;
  for (size_t i = 0; i < mCorrectors.size (); ++i) {
    double scale = mCorrectors[i]->correction (jet);
    jet *= scale;
    result *= scale;
  }
  return result;
}

/// apply correction using Jet information only
double ChainedJetCorrector::correction (const reco::Jet& fJet) const
{
  std::auto_ptr<reco::Jet> jet (dynamic_cast<reco::Jet*> (fJet.clone ()));
  double result = 1;
  for (size_t i = 0; i < mCorrectors.size (); ++i) {
    double scale = mCorrectors[i]->correction (*jet);
    jet->scaleEnergy (scale);
    result *= scale;
  }
  return result;
}

/// apply correction using all event information
double ChainedJetCorrector::correction (const reco::Jet& fJet,
					const edm::Event& fEvent,
					const edm::EventSetup& fSetup) const
{
  std::auto_ptr<reco::Jet> jet (dynamic_cast<reco::Jet*> (fJet.clone ()));
  double result = 1;
  for (size_t i = 0; i < mCorrectors.size (); ++i) {
    double scale = mCorrectors[i]->correction (*jet, fEvent, fSetup);
    jet->scaleEnergy (scale);
    result *= scale;
  }
  return result;
}
/// apply correction using all event information and reference to the raw jet
double ChainedJetCorrector::correction (const reco::Jet& fJet,
					const edm::RefToBase<reco::Jet>& fJetRef,
					const edm::Event& fEvent,
					const edm::EventSetup& fSetup) const
{
  std::auto_ptr<reco::Jet> jet (dynamic_cast<reco::Jet*> (fJet.clone ()));
  double result = 1;
  for (size_t i = 0; i < mCorrectors.size (); ++i) {
    double scale = mCorrectors[i]->correction (*jet, fJetRef, fEvent, fSetup);
    jet->scaleEnergy (scale);
    result *= scale;
  }
  return result;
}

/// if correction needs event information
bool ChainedJetCorrector::eventRequired () const
{
  for (size_t i = 0; i < mCorrectors.size (); ++i) {
    if (mCorrectors[i]->eventRequired ()) return true;
  }
  return false;
}

/// if correction needs jet reference
bool ChainedJetCorrector::refRequired () const
{
  for (size_t i = 0; i < mCorrectors.size (); ++i) {
    if (mCorrectors[i]->refRequired ()) return true;
  }
  return false;
}


