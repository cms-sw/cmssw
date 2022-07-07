/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToResResBuilderBase.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayGenericBuilderBase.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>
using namespace std;

//-------------------
// Initializations --
//-------------------

//----------------
// Constructors --
//----------------
BPHDecayToResResBuilderBase::BPHDecayToResResBuilderBase(
    const BPHEventSetupWrapper& es, const string& res1Name, double res1Mass, double res1Width, const string& res2Name)
    : BPHDecayGenericBuilderBase(es, nullptr),
      BPHDecayConstrainedBuilderBase(res1Name, res1Mass, res1Width),
      sName(res2Name),
      res2Sel(new BPHMassSelect(-2.0e+06, -1.0e+06)),
      dzFilter(&rName) {}

BPHDecayToResResBuilderBase::BPHDecayToResResBuilderBase(const string& res2Name)
    : sName(res2Name), res2Sel(new BPHMassSelect(-2.0e+06, -1.0e+06)), dzFilter(&rName) {}

//--------------
// Destructor --
//--------------
BPHDecayToResResBuilderBase::~BPHDecayToResResBuilderBase() { delete res2Sel; }

//--------------
// Operations --
//--------------

/// set cuts
void BPHDecayToResResBuilderBase::setRes2MassMin(double m) {
  outdated = true;
  res2Sel->setMassMin(m);
  return;
}

void BPHDecayToResResBuilderBase::setRes2MassMax(double m) {
  outdated = true;
  res2Sel->setMassMax(m);
  return;
}

void BPHDecayToResResBuilderBase::setRes2MassRange(double mMin, double mMax) {
  outdated = true;
  res2Sel->setMassMin(mMin);
  res2Sel->setMassMax(mMax);
  return;
}

/// build candidates
void BPHDecayToResResBuilderBase::fill(BPHRecoBuilder& brb, void* parameters) {
  brb.setMinPDiffererence(minPDiff);
  addRes1Collection(brb);
  addRes2Collection(brb);

  if (massSel->getMassMax() >= 0.0)
    brb.filter(*massSel);
  if (chi2Sel->getProbMin() >= 0.0)
    brb.filter(*chi2Sel);
  if (mFitSel->getMassMax() >= 0.0)
    brb.filter(*mFitSel);

  setup(parameters);

  return;
}
