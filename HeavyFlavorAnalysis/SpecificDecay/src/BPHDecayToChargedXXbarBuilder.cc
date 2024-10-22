/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToChargedXXbarBuilder.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayGenericBuilderBase.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticlePtSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleEtaSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMassSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHChi2Select.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleMasses.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHTrackReference.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Math/interface/LorentzVector.h"

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
BPHDecayToChargedXXbarBuilder::BPHDecayToChargedXXbarBuilder(const BPHEventSetupWrapper& es,
                                                             const string& dPosName,
                                                             const string& dNegName,
                                                             double daugMass,
                                                             double daugSigma,
                                                             const BPHRecoBuilder::BPHGenericCollection* posCollection,
                                                             const BPHRecoBuilder::BPHGenericCollection* negCollection)
    : BPHDecayGenericBuilderBase(es),
      ptMin(-1.0),
      etaMax(10.0),
      dzMax(1.0),
      pName(dPosName),
      nName(dNegName),
      dMass(daugMass),
      dSigma(daugSigma),
      pCollection(posCollection),
      nCollection(negCollection) {}

//--------------
// Operations --
//--------------

/// set cuts
void BPHDecayToChargedXXbarBuilder::setPtMin(double pt) {
  outdated = true;
  ptMin = pt;
  return;
}

void BPHDecayToChargedXXbarBuilder::setEtaMax(double eta) {
  outdated = true;
  etaMax = eta;
  return;
}

void BPHDecayToChargedXXbarBuilder::setDzMax(double dz) {
  outdated = true;
  dzMax = dz;
  return;
}

/// build candidates
void BPHDecayToChargedXXbarBuilder::fillRecList() {
  // extract basic informations from input collections

  vector<Particle*> pList;
  vector<Particle*> nList;

  addParticle(pCollection, +1, pList);
  addParticle(nCollection, -1, nList);
  int iPos;
  int iNeg;
  int nPos = pList.size();
  int nNeg = nList.size();
  double massMin = getMassMin();
  double massMax = getMassMax();
  double mSqMin = massMin * massMin * 0.9;
  double mSqMax = massMax * massMax * 1.1;
  if (mSqMin < 0.0)
    mSqMin = 0.0;

  for (iPos = 0; iPos < nPos; ++iPos) {
    Particle* pc = pList[iPos];
    const reco::Track* pt = pc->track;
    double px = pc->px;
    double py = pc->py;
    double pz = pc->pz;
    double pe = pc->en;
    for (iNeg = 0; iNeg < nNeg; ++iNeg) {
      Particle* nc = nList[iNeg];
      const reco::Track* nt = nc->track;
      if (fabs(nt->dz() - pt->dz()) > 1.0)
        continue;
      double nx = nc->px;
      double ny = nc->py;
      double nz = nc->pz;
      double ne = nc->en;
      const float tx = px + nx;
      const float ty = py + ny;
      const float tz = pz + nz;
      const float te = pe + ne;
      const float m2 = (te * te) - ((tx * tx) + (ty * ty) + (tz * tz));
      if (m2 < mSqMin)
        continue;
      if (m2 > mSqMax)
        continue;
      BPHPlusMinusCandidatePtr rc = BPHPlusMinusCandidateWrap::create(evSetup);
      rc->add(pName, pc->cand, dMass, dSigma);
      rc->add(nName, nc->cand, dMass, dSigma);
      double mass = rc->composite().mass();
      if (mass < massMin)
        continue;
      if (mass > massMax)
        continue;
      if (!chi2Sel->accept(*rc))
        continue;
      recList.push_back(rc);
    }
  }

  for (iPos = 0; iPos < nPos; ++iPos)
    delete pList[iPos];
  for (iNeg = 0; iNeg < nNeg; ++iNeg)
    delete nList[iNeg];

  return;
}

void BPHDecayToChargedXXbarBuilder::addParticle(const BPHRecoBuilder::BPHGenericCollection* collection,
                                                int charge,
                                                vector<Particle*>& list) {
  int i;
  int n = collection->size();
  list.reserve(n);
  for (i = 0; i < n; ++i) {
    const reco::Candidate& cand = collection->get(i);
    int q = cand.charge();
    if ((charge > 0) && (q <= 0))
      continue;
    if ((charge < 0) && (q >= 0))
      continue;
    const reco::Candidate::LorentzVector p4 = cand.p4();
    if (p4.pt() < ptMin)
      continue;
    if (p4.eta() > etaMax)
      continue;
    const reco::Track* tk = BPHTrackReference::getTrack(cand, "cfhp");
    if (tk == nullptr)
      continue;
    double px = p4.px();
    double py = p4.py();
    double pz = p4.pz();
    list.push_back(new Particle(&cand, tk, px, py, pz, sqrt((px * px) + (py * py) + (pz * pz) + (dMass * dMass))));
  }
  return;
}
