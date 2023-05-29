/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayConstrainedBuilderBase.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHAnalyzerTokenWrapper.h"

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
BPHDecayConstrainedBuilderBase::BPHDecayConstrainedBuilderBase(const BPHEventSetupWrapper& es,
                                                               const string& resName,
                                                               double resMass,
                                                               double resWidth)
    : BPHDecayConstrainedBuilderBase(resName, resMass, resWidth) {
  if (evSetup == nullptr)
    evSetup = new BPHEventSetupWrapper(es);
}

BPHDecayConstrainedBuilderBase::BPHDecayConstrainedBuilderBase(const string& resName,
                                                               double resMass,
                                                               double resWidth,
                                                               bool createFitSelObject)
    : rName(resName),
      rMass(resMass),
      rWidth(resWidth),
      resoSel(new BPHMassSelect(-2.0e+06, -1.0e+06)),
      massConstr(true),
      mfSelForce(false) {
  mFitSel = (createFitSelObject ? massFitSelector(-2.0e+06, -1.0e+06) : nullptr);
}

BPHDecayConstrainedBuilderBase::BPHDecayConstrainedBuilderBase() {}

//--------------
// Destructor --
//--------------
BPHDecayConstrainedBuilderBase::~BPHDecayConstrainedBuilderBase() { delete resoSel; }

//--------------
// Operations --
//--------------
/// set cuts
void BPHDecayConstrainedBuilderBase::setResMassMin(double m) {
  outdated = true;
  resoSel->setMassMin(m);
  return;
}

void BPHDecayConstrainedBuilderBase::setResMassMax(double m) {
  outdated = true;
  resoSel->setMassMax(m);
  return;
}

void BPHDecayConstrainedBuilderBase::setResMassRange(double mMin, double mMax) {
  outdated = true;
  resoSel->setMassMin(mMin);
  resoSel->setMassMax(mMax);
  return;
}

void BPHDecayConstrainedBuilderBase::setConstr(bool flag) {
  if ((flag == massConstr) && !mfSelForce)
    return;
  outdated = true;
  massConstr = flag;
  BPHMassFitSelect* mfs = massFitSelector(mFitSel->getMassMin(), mFitSel->getMassMax());
  delete mFitSel;
  mFitSel = mfs;
  mfSelForce = false;
  return;
}

void BPHDecayConstrainedBuilderBase::setMassFitSelect(BPHMassFitSelect* mfs) {
  if (mFitSel == mfs)
    return;
  outdated = true;
  mfSelForce = true;
  mFitSel = mfs;
  return;
}

BPHMassFitSelect* BPHDecayConstrainedBuilderBase::massFitSelector(double mMin, double mMax) {
  if (massConstr)
    return new BPHMassFitSelect(rName, rMass, rWidth, mMin, mMax);
  else
    return new BPHMassFitSelect(mMin, mMax);
}
