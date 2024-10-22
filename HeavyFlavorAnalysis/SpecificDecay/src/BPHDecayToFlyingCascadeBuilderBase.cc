/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToFlyingCascadeBuilderBase.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayGenericBuilderBase.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
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
BPHDecayToFlyingCascadeBuilderBase::BPHDecayToFlyingCascadeBuilderBase(const BPHEventSetupWrapper& es,
                                                                       const string& flyName,
                                                                       double flyMass,
                                                                       double flyMSigma)
    : BPHDecayToFlyingCascadeBuilderBase(flyName, flyMass, flyMSigma) {
  if (evSetup == nullptr)
    evSetup = new BPHEventSetupWrapper(es);
}

BPHDecayToFlyingCascadeBuilderBase::BPHDecayToFlyingCascadeBuilderBase(const string& flyName,
                                                                       double flyMass,
                                                                       double flyMSigma)
    : fName(flyName),
      fMass(flyMass),
      fMSigma(flyMSigma),
      flySel(new BPHMassFitSelect(-2.0e+06, -1.0e+06)),
      kfChi2Sel(new BPHKinFitChi2Select(-1.0)) {}

BPHDecayToFlyingCascadeBuilderBase::BPHDecayToFlyingCascadeBuilderBase()
    : flySel(new BPHMassFitSelect(-2.0e+06, -1.0e+06)), kfChi2Sel(new BPHKinFitChi2Select(-1.0)) {}

//--------------
// Destructor --
//--------------
BPHDecayToFlyingCascadeBuilderBase::~BPHDecayToFlyingCascadeBuilderBase() {
  delete flySel;
  delete kfChi2Sel;
}

//--------------
// Operations --
//--------------

/// set cuts
void BPHDecayToFlyingCascadeBuilderBase::setFlyingMassMin(double m) {
  outdated = true;
  flySel->setMassMin(m);
  return;
}

void BPHDecayToFlyingCascadeBuilderBase::setFlyingMassMax(double m) {
  outdated = true;
  flySel->setMassMax(m);
  return;
}

void BPHDecayToFlyingCascadeBuilderBase::setFlyingMassRange(double mMin, double mMax) {
  outdated = true;
  flySel->setMassMin(mMin);
  flySel->setMassMax(mMax);
  return;
}

void BPHDecayToFlyingCascadeBuilderBase::setKinFitProbMin(double p) {
  outdated = true;
  kfChi2Sel->setProbMin(p);
}
