/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToResResBuilder.h"

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
BPHDecayToResResBuilder::BPHDecayToResResBuilder(const edm::EventSetup& es,
                                                 const std::string& res1Name,
                                                 double res1Mass,
                                                 double res1Width,
                                                 const std::vector<BPHPlusMinusConstCandPtr>& res1Collection,
                                                 const std::string& res2Name,
                                                 const std::vector<BPHPlusMinusConstCandPtr>& res2Collection)
    : BPHDecayConstrainedBuilder(es, res1Name, res1Mass, res1Width, res1Collection),
      sName(res2Name),
      sCollection(&res2Collection),
      res2Sel(new BPHMassSelect(-2.0e+06, -1.0e+06)) {}

//--------------
// Destructor --
//--------------
BPHDecayToResResBuilder::~BPHDecayToResResBuilder() { delete res2Sel; }

//--------------
// Operations --
//--------------
vector<BPHRecoConstCandPtr> BPHDecayToResResBuilder::build() {
  if (updated)
    return recList;

  recList.clear();

  BPHRecoBuilder brb(*evSetup);
  brb.setMinPDiffererence(minPDiff);
  brb.add(rName, *rCollection);
  brb.add(sName, *sCollection);
  brb.filter(rName, *resoSel);
  brb.filter(sName, *res2Sel);

  if (massSel->getMassMax() >= 0.0)
    brb.filter(*massSel);
  if (chi2Sel->getProbMin() >= 0.0)
    brb.filter(*chi2Sel);
  if (mFitSel->getMassMax() >= 0.0)
    brb.filter(*mFitSel);

  recList = BPHRecoCandidate::build(brb);
  updated = true;
  return recList;
}

/// set cuts
void BPHDecayToResResBuilder::setRes2MassMin(double m) {
  updated = false;
  res2Sel->setMassMin(m);
  return;
}

void BPHDecayToResResBuilder::setRes2MassMax(double m) {
  updated = false;
  res2Sel->setMassMax(m);
  return;
}

void BPHDecayToResResBuilder::setRes2MassRange(double mMin, double mMax) {
  updated = false;
  res2Sel->setMassMin(mMin);
  res2Sel->setMassMax(mMax);
  return;
}
