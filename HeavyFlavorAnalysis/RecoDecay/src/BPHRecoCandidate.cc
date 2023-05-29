/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHAnalyzerTokenWrapper.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"

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
BPHRecoCandidate::BPHRecoCandidate(const BPHEventSetupWrapper* es, int daugNum, int compNum)
    : BPHDecayMomentum(daugNum, compNum), BPHDecayVertex(es), BPHKinematicFit() {}

BPHRecoCandidate::BPHRecoCandidate(const BPHEventSetupWrapper* es, const BPHRecoBuilder::ComponentSet& compList)
    : BPHDecayMomentum(compList.daugMap, compList.compMap), BPHDecayVertex(this, es), BPHKinematicFit(this) {}

//--------------
// Operations --
//--------------
vector<BPHRecoConstCandPtr> BPHRecoCandidate::build(const BPHRecoBuilder& builder, double mass, double msig) {
  // create a list of pointers to BPHRecoCandidate and fill it
  // with particle combinations selected by the BPHRecoBuilder
  vector<BPHRecoConstCandPtr> cList;
  fill<BPHRecoCandidate>(cList, builder, mass, msig);
  return cList;
}

/// clone object, cloning daughters as well up to required depth
/// level = -1 to clone all levels
BPHRecoCandidate* BPHRecoCandidate::clone(int level) const {
  BPHRecoCandidate* ptr = new BPHRecoCandidate(getEventSetup());
  fill(ptr, level);
  return ptr;
}

// function doing the job to clone reconstructed decays:
// copy stable particles and clone cascade decays up to chosen level
void BPHRecoCandidate::fill(BPHRecoCandidate* ptr, int level) const {
  ptr->setConstraint(constrMass(), constrSigma());
  const vector<string>& nDaug = daugNames();
  int id;
  int nd = nDaug.size();
  for (id = 0; id < nd; ++id) {
    const string& n = nDaug[id];
    const reco::Candidate* d = getDaug(n);
    ptr->add(n, originalReco(d), getTrackSearchList(d), d->mass(), getMassSigma(d));
  }
  const vector<string>& nComp = compNames();
  int ic;
  int nc = nComp.size();
  for (ic = 0; ic < nc; ++ic) {
    const string& n = nComp[ic];
    BPHRecoConstCandPtr c = getComp(n);
    if (level)
      ptr->add(n, BPHRecoConstCandPtr(c->clone(level - 1)));
    else
      ptr->add(n, c);
    double m = -1;
    double s = -1;
    if (getIndependentFit(n, m, s))
      ptr->setIndependentFit(n, true, m, s);
  }
  return;
}
