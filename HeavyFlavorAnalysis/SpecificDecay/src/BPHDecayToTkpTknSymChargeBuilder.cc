/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToTkpTknSymChargeBuilder.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticlePtSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleEtaSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMassSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMassSymSelect.h"
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
BPHDecayToTkpTknSymChargeBuilder::BPHDecayToTkpTknSymChargeBuilder(
    const BPHEventSetupWrapper& es,
    const string& daug1Name,
    double daug1Mass,
    double daug1Sigma,
    const string& daug2Name,
    double daug2Mass,
    double daug2Sigma,
    const BPHRecoBuilder::BPHGenericCollection* posCollection,
    const BPHRecoBuilder::BPHGenericCollection* negCollection,
    double expectedMass)
    : BPHDecayGenericBuilderBase(es),
      d1Name(daug1Name),
      d1Mass(daug1Mass),
      d1Sigma(daug1Sigma),
      d2Name(daug2Name),
      d2Mass(daug2Mass),
      d2Sigma(daug2Sigma),
      eMass(expectedMass),
      pCollection(posCollection),
      nCollection(negCollection),
      pt1Min(-1.0),
      pt2Min(-1.0),
      eta1Max(10.0),
      eta2Max(10.0),
      dzMax(1.0) {}

//--------------
// Operations --
//--------------
void BPHDecayToTkpTknSymChargeBuilder::fillRecList() {
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
    double p1 = pc->e1;
    double p2 = pc->e2;
    for (iNeg = 0; iNeg < nNeg; ++iNeg) {
      Particle* nc = nList[iNeg];
      const reco::Track* nt = nc->track;
      if (fabs(nt->dz() - pt->dz()) > dzMax)
        continue;
      double nx = nc->px;
      double ny = nc->py;
      double nz = nc->pz;
      double n1 = nc->e1;
      double n2 = nc->e2;
      const float tx = px + nx;
      const float ty = py + ny;
      const float tz = pz + nz;
      const float ta = ((p1 > 0.0) && (n2 > 0.0) ? p1 + n2 : -1.0);
      const float tb = ((p2 > 0.0) && (n1 > 0.0) ? p2 + n1 : -1.0);
      float ma = (ta > 0 ? (ta * ta) - ((tx * tx) + (ty * ty) + (tz * tz)) : -1.0);
      float mb = (tb > 0 ? (tb * tb) - ((tx * tx) + (ty * ty) + (tz * tz)) : -1.0);
      if (((ma < mSqMin) || (ma > mSqMax)) && ((mb < mSqMin) || (mb > mSqMax)))
        continue;
      BPHPlusMinusCandidatePtr rc(nullptr);
      float rcMass = -1.0;
      if (ma > 0.0) {
        rc = BPHPlusMinusCandidateWrap::create(evSetup);
        rc->add(d1Name, pc->cand, d1Mass, d1Sigma);
        rc->add(d2Name, nc->cand, d2Mass, d2Sigma);
        rcMass = rc->composite().mass();
      }
      BPHPlusMinusCandidatePtr rb(nullptr);
      float rbMass = -1.0;
      if (mb > 0.0) {
        rb = BPHPlusMinusCandidateWrap::create(evSetup);
        rb->add(d1Name, nc->cand, d1Mass, d1Sigma);
        rb->add(d2Name, pc->cand, d2Mass, d2Sigma);
        rbMass = rb->composite().mass();
      }
      BPHPlusMinusCandidatePtr* rp(nullptr);
      double mass = -1.0;
      if ((rc.get() != nullptr) && ((rb.get() == nullptr) || (fabs(rcMass - eMass) < fabs(rbMass - eMass)))) {
        mass = rcMass;
        rp = &rc;
      } else {
        mass = rbMass;
        rp = &rb;
      }
      BPHPlusMinusCandidate* rr = rp->get();
      if (mass < massMin)
        continue;
      if (mass > massMax)
        continue;
      if (!chi2Sel->accept(*rr))
        continue;
      recList.push_back(*rp);
    }
  }

  for (iPos = 0; iPos < nPos; ++iPos)
    delete pList[iPos];
  for (iNeg = 0; iNeg < nNeg; ++iNeg)
    delete nList[iNeg];

  return;
}

/// set cuts
void BPHDecayToTkpTknSymChargeBuilder::setTrk1PtMin(double pt) {
  outdated = true;
  pt1Min = pt;
  return;
}

void BPHDecayToTkpTknSymChargeBuilder::setTrk2PtMin(double pt) {
  outdated = true;
  pt2Min = pt;
  return;
}

void BPHDecayToTkpTknSymChargeBuilder::setTrk1EtaMax(double eta) {
  outdated = true;
  eta1Max = eta;
  return;
}

void BPHDecayToTkpTknSymChargeBuilder::setTrk2EtaMax(double eta) {
  outdated = true;
  eta2Max = eta;
  return;
}

void BPHDecayToTkpTknSymChargeBuilder::setDzMax(double dz) {
  outdated = true;
  dzMax = dz;
  return;
}

void BPHDecayToTkpTknSymChargeBuilder::addParticle(const BPHRecoBuilder::BPHGenericCollection* collection,
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
    const reco::Track* tk = BPHTrackReference::getTrack(cand, "cfhp");
    if (tk == nullptr)
      continue;
    double px = p4.px();
    double py = p4.py();
    double pz = p4.pz();
    double p2 = (px * px) + (py * py) + (pz * pz);
    double e1 = ((p4.pt() >= pt1Min) && (p4.eta() <= eta1Max) ? sqrt(p2 + (d1Mass * d1Mass)) : -1.0);
    double e2 = ((p4.pt() >= pt2Min) && (p4.eta() <= eta2Max) ? sqrt(p2 + (d2Mass * d2Mass)) : -1.0);
    if ((e1 > 0.0) || (e2 > 0.0))
      list.push_back(new Particle(&cand, tk, px, py, pz, e1, e2));
  }
  return;
}
