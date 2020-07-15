/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayGenericBuilder.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

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
BPHDecayGenericBuilder::BPHDecayGenericBuilder(const edm::EventSetup& es, BPHMassFitSelect* mfs)
    : evSetup(&es),
      massSel(new BPHMassSelect(-2.0e+06, -1.0e+06)),
      chi2Sel(new BPHChi2Select(-1.0)),
      mFitSel(mfs),
      minPDiff(1.0e-4),
      updated(false) {
  if (mFitSel == nullptr)
    mFitSel = new BPHMassFitSelect(-2.0e+06, -1.0e+06);
}

//--------------
// Destructor --
//--------------
BPHDecayGenericBuilder::~BPHDecayGenericBuilder() {
  delete massSel;
  delete chi2Sel;
  delete mFitSel;
}

//--------------
// Operations --
//--------------
/// set cuts
void BPHDecayGenericBuilder::setMassMin(double m) {
  updated = false;
  massSel->setMassMin(m);
  return;
}

void BPHDecayGenericBuilder::setMassMax(double m) {
  updated = false;
  massSel->setMassMax(m);
  return;
}

void BPHDecayGenericBuilder::setMassRange(double mMin, double mMax) {
  updated = false;
  massSel->setMassMin(mMin);
  massSel->setMassMax(mMax);
  return;
}

void BPHDecayGenericBuilder::setProbMin(double p) {
  updated = false;
  chi2Sel->setProbMin(p);
  return;
}

void BPHDecayGenericBuilder::setMassFitMin(double m) {
  updated = false;
  mFitSel->setMassMin(m);
  return;
}

void BPHDecayGenericBuilder::setMassFitMax(double m) {
  updated = false;
  mFitSel->setMassMax(m);
  return;
}

void BPHDecayGenericBuilder::setMassFitRange(double mMin, double mMax) {
  updated = false;
  mFitSel->setMassMin(mMin);
  mFitSel->setMassMax(mMax);
  return;
}
