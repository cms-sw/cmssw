//
// Original Author:  Fedor Ratnikov Feb. 16, 2007
// $Id: ChainedJetCorrector.cc,v 1.1 2008/02/29 23:00:36 fedor Exp $
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
