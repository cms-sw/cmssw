/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayConstrainedBuilder.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"

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
BPHDecayConstrainedBuilder::BPHDecayConstrainedBuilder(const edm::EventSetup& es,
                                                       const std::string& resName,
                                                       double resMass,
                                                       double resWidth,
                                                       const std::vector<BPHPlusMinusConstCandPtr>& resCollection)
    : BPHDecayGenericBuilder(es, new BPHMassFitSelect(resName, resMass, resWidth, -2.0e+06, -1.0e+06)),
      rName(resName),
      rMass(resMass),
      rWidth(resWidth),
      rCollection(&resCollection),
      resoSel(new BPHMassSelect(-2.0e+06, -1.0e+06)),
      massConstr(true) {}

//--------------
// Destructor --
//--------------
BPHDecayConstrainedBuilder::~BPHDecayConstrainedBuilder() { delete resoSel; }

//--------------
// Operations --
//--------------
/// set cuts
void BPHDecayConstrainedBuilder::setResMassMin(double m) {
  updated = false;
  resoSel->setMassMin(m);
  return;
}

void BPHDecayConstrainedBuilder::setResMassMax(double m) {
  updated = false;
  resoSel->setMassMax(m);
  return;
}

void BPHDecayConstrainedBuilder::setResMassRange(double mMin, double mMax) {
  updated = false;
  resoSel->setMassMin(mMin);
  resoSel->setMassMax(mMax);
  return;
}

void BPHDecayConstrainedBuilder::setConstr(bool flag) {
  updated = false;
  if (flag == massConstr)
    return;
  double mMin = mFitSel->getMassMin();
  double mMax = mFitSel->getMassMax();
  delete mFitSel;
  massConstr = flag;
  if (massConstr)
    mFitSel = new BPHMassFitSelect(rName, rMass, rWidth, mMin, mMax);
  else
    mFitSel = new BPHMassFitSelect(mMin, mMax);
  return;
}
