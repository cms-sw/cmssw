/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToResFlyingBuilder.h"

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
BPHDecayToResFlyingBuilder::BPHDecayToResFlyingBuilder(const edm::EventSetup& es,
                                                       const std::string& resName,
                                                       double resMass,
                                                       double resWidth,
                                                       const std::vector<BPHPlusMinusConstCandPtr>& resCollection,
                                                       const std::string& flyName,
                                                       double flyMass,
                                                       double flyMSigma,
                                                       const std::vector<BPHPlusMinusConstCandPtr>& flyCollection)
    : BPHDecayConstrainedBuilder(es, resName, resMass, resWidth, resCollection),
      fName(flyName),
      fMass(flyMass),
      fMSigma(flyMSigma),
      fCollection(&flyCollection),
      flySel(new BPHMassFitSelect(-2.0e+06, -1.0e+06)),
      kfChi2Sel(new BPHKinFitChi2Select(-1.0)) {}

//--------------
// Destructor --
//--------------
BPHDecayToResFlyingBuilder::~BPHDecayToResFlyingBuilder() {
  delete flySel;
  delete kfChi2Sel;
}

//--------------
// Operations --
//--------------
vector<BPHRecoConstCandPtr> BPHDecayToResFlyingBuilder::build() {
  if (updated)
    return recList;

  recList.clear();

  BPHRecoBuilder brb(*evSetup);
  brb.setMinPDiffererence(minPDiff);
  brb.add(rName, *rCollection);
  brb.add(fName, *fCollection);

  if (resoSel->getMassMax() >= 0.0)
    brb.filter(rName, *resoSel);
  if (flySel->getMassMax() >= 0.0)
    brb.filter(fName, *flySel);

  if (massSel->getMassMax() >= 0.0)
    brb.filter(*massSel);

  vector<BPHRecoConstCandPtr> tmpList = BPHRecoCandidate::build(brb);
  //
  //  Apply kinematic constraint on the resonance mass.
  //
  int iRec;
  int nRec = tmpList.size();
  recList.reserve(nRec);
  for (iRec = 0; iRec < nRec; ++iRec) {
    BPHRecoConstCandPtr ctmp = tmpList[iRec];
    BPHRecoCandidate* cptr = ctmp->clone();
    BPHRecoConstCandPtr cand(cptr);
    // fit for flying reconstruction
    // indipendent from other particles
    cptr->setIndependentFit(fName, true, fMass, fMSigma);
    cptr->resetKinematicFit();
    if ((mFitSel->getMassMax() >= 0) && (!mFitSel->accept(*cptr)))
      continue;
    const RefCountedKinematicVertex tdv = cptr->topDecayVertex();
    if ((kfChi2Sel->getProbMin() >= 0) && !kfChi2Sel->accept(*cptr))
      continue;
    dMap[cand->getComp(rName).get()] = ctmp->getComp(rName).get();
    dMap[cand->getComp(fName).get()] = ctmp->getComp(fName).get();
    recList.push_back(cand);
  }
  updated = true;
  return recList;
}

/// set cuts
void BPHDecayToResFlyingBuilder::setFlyingMassMin(double m) {
  updated = false;
  flySel->setMassMin(m);
  return;
}

void BPHDecayToResFlyingBuilder::setFlyingMassMax(double m) {
  updated = false;
  flySel->setMassMax(m);
  return;
}

void BPHDecayToResFlyingBuilder::setFlyingMassRange(double mMin, double mMax) {
  updated = false;
  flySel->setMassMin(mMin);
  flySel->setMassMax(mMax);
  return;
}

void BPHDecayToResFlyingBuilder::setKinFitProbMin(double p) {
  updated = false;
  kfChi2Sel->setProbMin(p);
}
