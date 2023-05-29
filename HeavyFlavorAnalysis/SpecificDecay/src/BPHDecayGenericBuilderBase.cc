/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayGenericBuilderBase.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHAnalyzerTokenWrapper.h"

//---------------
// C++ Headers --
//---------------
using namespace std;

//-------------------
// Initializations --
//-------------------

//----------------
// Constructors --
//----------------
BPHDecayGenericBuilderBase::BPHDecayGenericBuilderBase()
    : evSetup(nullptr),
      massSel(new BPHMassSelect(-2.0e+06, -1.0e+06)),
      chi2Sel(new BPHChi2Select(-1.0)),
      mFitSel(nullptr),
      minPDiff(1.0e-4),
      outdated(true) {}

BPHDecayGenericBuilderBase::BPHDecayGenericBuilderBase(const BPHEventSetupWrapper& es, BPHMassFitSelect* mfs)
    : BPHDecayGenericBuilderBase() {
  evSetup = new BPHEventSetupWrapper(es);
  mFitSel = mfs;
}

//--------------
// Destructor --
//--------------
BPHDecayGenericBuilderBase::~BPHDecayGenericBuilderBase() {
  delete massSel;
  delete chi2Sel;
  delete mFitSel;
  delete evSetup;
}

//--------------
// Operations --
//--------------
/// set cuts
void BPHDecayGenericBuilderBase::setMassMin(double m) {
  outdated = true;
  massSel->setMassMin(m);
  return;
}

void BPHDecayGenericBuilderBase::setMassMax(double m) {
  outdated = true;
  massSel->setMassMax(m);
  return;
}

void BPHDecayGenericBuilderBase::setMassRange(double mMin, double mMax) {
  outdated = true;
  massSel->setMassMin(mMin);
  massSel->setMassMax(mMax);
  return;
}

void BPHDecayGenericBuilderBase::setProbMin(double p) {
  outdated = true;
  chi2Sel->setProbMin(p);
  return;
}

void BPHDecayGenericBuilderBase::setMassFitMin(double m) {
  outdated = true;
  mFitSel->setMassMin(m);
  return;
}

void BPHDecayGenericBuilderBase::setMassFitMax(double m) {
  outdated = true;
  mFitSel->setMassMax(m);
  return;
}

void BPHDecayGenericBuilderBase::setMassFitRange(double mMin, double mMax) {
  outdated = true;
  mFitSel->setMassMin(mMin);
  mFitSel->setMassMax(mMax);
  return;
}
