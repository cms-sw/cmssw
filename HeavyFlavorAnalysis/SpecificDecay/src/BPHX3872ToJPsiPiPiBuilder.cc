/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHX3872ToJPsiPiPiBuilder.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHTrackReference.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMassSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHChi2Select.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMassFitSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleMasses.h"

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
BPHX3872ToJPsiPiPiBuilder::BPHX3872ToJPsiPiPiBuilder(const edm::EventSetup& es,
                                                     const std::vector<BPHPlusMinusConstCandPtr>& jpsiCollection,
                                                     const BPHRecoBuilder::BPHGenericCollection* posCollection,
                                                     const BPHRecoBuilder::BPHGenericCollection* negCollection)
    : jPsiName("JPsi"),
      pionPosName("PionPos"),
      pionNegName("PionNeg"),
      evSetup(&es),
      jCollection(&jpsiCollection),
      pCollection(posCollection),
      nCollection(negCollection) {
  jpsiSel = new BPHMassSelect(2.80, 3.40);
  ptMin = 1.0;
  etaMax = 10.0;
  massSel = new BPHMassSelect(3.00, 4.50);
  chi2Sel = new BPHChi2Select(0.02);
  mFitSel = new BPHMassFitSelect(jPsiName, BPHParticleMasses::jPsiMass, BPHParticleMasses::jPsiMWidth, 3.50, 4.20);
  massConstr = true;
  minPDiff = 1.0e-4;
  updated = false;
}

//--------------
// Destructor --
//--------------
BPHX3872ToJPsiPiPiBuilder::~BPHX3872ToJPsiPiPiBuilder() {
  delete jpsiSel;
  delete massSel;
  delete chi2Sel;
  delete mFitSel;
}

//--------------
// Operations --
//--------------
vector<BPHRecoConstCandPtr> BPHX3872ToJPsiPiPiBuilder::build() {
  if (updated)
    return x3872List;

  x3872List.clear();

  // extract basic informations from input collections

  class Particle {
  public:
    Particle(const reco::Candidate* c, const reco::Track* tk, double x, double y, double z, double p)
        : cand(c), track(tk), px(x), py(y), pz(z), ePion(p) {}
    const reco::Candidate* cand;
    const reco::Track* track;
    double px;
    double py;
    double pz;
    double ePion;
  };

  vector<Particle> pList;
  vector<Particle> nList;

  int nPos = pCollection->size();
  int nNeg = nCollection->size();

  pList.reserve(nPos);
  nList.reserve(nNeg);

  // filter input collections

  int iPos;
  int iNeg;

  for (iPos = 0; iPos < nPos; ++iPos) {
    const reco::Candidate& cand = pCollection->get(iPos);
    if (cand.charge() != +1)
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
    double p2 = (px * px) + (py * py) + (pz * pz);
    pList.push_back(
        Particle(&cand, tk, px, py, pz, sqrt(p2 + (BPHParticleMasses::pionMass * BPHParticleMasses::pionMass))));
  }

  for (iNeg = 0; iNeg < nNeg; ++iNeg) {
    const reco::Candidate& cand = nCollection->get(iNeg);
    if (cand.charge() != -1)
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
    double p2 = (px * px) + (py * py) + (pz * pz);
    nList.push_back(
        Particle(&cand, tk, px, py, pz, sqrt(p2 + (BPHParticleMasses::pionMass * BPHParticleMasses::pionMass))));
  }

  // filter basic candidates

  nPos = pList.size();
  nNeg = nList.size();

  double mMax = getMassMax() - (0.8 * BPHParticleMasses::jPsiMass);

  struct PionPair {
    const reco::Candidate* posPion;
    const reco::Candidate* negPion;
  };
  vector<const PionPair*> pionPairs;
  pionPairs.reserve(nPos * nNeg);

  for (iPos = 0; iPos < nPos; ++iPos) {
    Particle& pc = pList[iPos];
    const reco::Track* pt = pc.track;
    double px = pc.px;
    double py = pc.py;
    double pz = pc.pz;
    double pe = pc.ePion;
    for (iNeg = 0; iNeg < nNeg; ++iNeg) {
      Particle& nc = nList[iNeg];
      const reco::Track* nt = nc.track;
      if (fabs(nt->dz() - pt->dz()) > 1.0)
        continue;
      double nx = nc.px;
      double ny = nc.py;
      double nz = nc.pz;
      double ne = nc.ePion;
      const float tx = px + nx;
      const float ty = py + ny;
      const float tz = pz + nz;
      const float te = pe + ne;
      float mass = (te * te) - ((tx * tx) + (ty * ty) + (tz * tz));
      if (mass > mMax)
        continue;
      PionPair* pp = new PionPair;
      pp->posPion = pc.cand;
      pp->negPion = nc.cand;
      pionPairs.push_back(pp);
    }
  }

  vector<BPHPlusMinusConstCandPtr> jPsi;
  int nJPsi = jCollection->size();
  int iJPsi;
  jPsi.reserve(nJPsi);
  for (iJPsi = 0; iJPsi < nJPsi; ++iJPsi) {
    const BPHPlusMinusConstCandPtr& jpCand = jCollection->at(iJPsi);
    if (jpsiSel->accept(*jpCand))
      jPsi.push_back(jpCand);
  }
  nJPsi = jPsi.size();

  int nPair = pionPairs.size();
  int iPair;
  for (iPair = 0; iPair < nPair; ++iPair) {
    const PionPair* pp = pionPairs[iPair];
    for (iJPsi = 0; iJPsi < nJPsi; ++iJPsi) {
      BPHRecoCandidate* x3872 = new BPHRecoCandidate(evSetup);
      BPHRecoCandidatePtr xPtr(x3872);
      x3872->add(jPsiName, jPsi[iJPsi]);
      x3872->add(pionPosName, pp->posPion, BPHParticleMasses::pionMass, BPHParticleMasses::pionMSigma);
      x3872->add(pionNegName, pp->negPion, BPHParticleMasses::pionMass, BPHParticleMasses::pionMSigma);
      if (!massSel->accept(*x3872))
        continue;
      if ((chi2Sel != nullptr) && !chi2Sel->accept(*x3872))
        continue;
      if (!mFitSel->accept(*x3872))
        continue;
      x3872List.push_back(xPtr);
    }
    delete pp;
  }

  updated = true;
  return x3872List;
}

/// set cuts
void BPHX3872ToJPsiPiPiBuilder::setJPsiMassMin(double m) {
  updated = false;
  jpsiSel->setMassMin(m);
  return;
}

void BPHX3872ToJPsiPiPiBuilder::setJPsiMassMax(double m) {
  updated = false;
  jpsiSel->setMassMax(m);
  return;
}

void BPHX3872ToJPsiPiPiBuilder::setPiPtMin(double pt) {
  updated = false;
  ptMin = pt;
  return;
}

void BPHX3872ToJPsiPiPiBuilder::setPiEtaMax(double eta) {
  updated = false;
  etaMax = eta;
  return;
}

void BPHX3872ToJPsiPiPiBuilder::setMassMin(double m) {
  updated = false;
  massSel->setMassMin(m);
  return;
}

void BPHX3872ToJPsiPiPiBuilder::setMassMax(double m) {
  updated = false;
  massSel->setMassMax(m);
  return;
}

void BPHX3872ToJPsiPiPiBuilder::setProbMin(double p) {
  updated = false;
  delete chi2Sel;
  chi2Sel = (p < 0.0 ? nullptr : new BPHChi2Select(p));
  return;
}

void BPHX3872ToJPsiPiPiBuilder::setMassFitMin(double m) {
  updated = false;
  mFitSel->setMassMin(m);
  return;
}

void BPHX3872ToJPsiPiPiBuilder::setMassFitMax(double m) {
  updated = false;
  mFitSel->setMassMax(m);
  return;
}

void BPHX3872ToJPsiPiPiBuilder::setConstr(bool flag) {
  updated = false;
  massConstr = flag;
  return;
}

/// get current cuts
double BPHX3872ToJPsiPiPiBuilder::getJPsiMassMin() const { return jpsiSel->getMassMin(); }

double BPHX3872ToJPsiPiPiBuilder::getJPsiMassMax() const { return jpsiSel->getMassMax(); }

double BPHX3872ToJPsiPiPiBuilder::getPiPtMin() const { return ptMin; }

double BPHX3872ToJPsiPiPiBuilder::getPiEtaMax() const { return etaMax; }

double BPHX3872ToJPsiPiPiBuilder::getMassMin() const { return massSel->getMassMin(); }

double BPHX3872ToJPsiPiPiBuilder::getMassMax() const { return massSel->getMassMax(); }

double BPHX3872ToJPsiPiPiBuilder::getProbMin() const { return (chi2Sel == nullptr ? -1.0 : chi2Sel->getProbMin()); }

double BPHX3872ToJPsiPiPiBuilder::getMassFitMin() const { return mFitSel->getMassMin(); }

double BPHX3872ToJPsiPiPiBuilder::getMassFitMax() const { return mFitSel->getMassMax(); }

bool BPHX3872ToJPsiPiPiBuilder::getConstr() const { return massConstr; }
