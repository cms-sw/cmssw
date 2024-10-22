/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHOniaToMuMuBuilder.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMuonPtSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMuonEtaSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMassSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHChi2Select.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleMasses.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHMultiSelect.h"

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
BPHOniaToMuMuBuilder::BPHOniaToMuMuBuilder(const BPHEventSetupWrapper& es,
                                           const BPHRecoBuilder::BPHGenericCollection* muPosCollection,
                                           const BPHRecoBuilder::BPHGenericCollection* muNegCollection)
    : BPHDecayGenericBuilderBase(es),
      muPosName("MuPos"),
      muNegName("MuNeg"),
      posCollection(muPosCollection),
      negCollection(muNegCollection) {
  setParameters(NRes, 2.0, 10.0, 0.01, 50.00, 2.0, -1.0, 0.0);
  setParameters(Phi, 2.0, 10.0, 0.50, 1.50, 0.0, BPHParticleMasses::phiMass, BPHParticleMasses::phiMWidth);
  setParameters(Psi1, 2.0, 10.0, 2.00, 3.40, 0.0, BPHParticleMasses::jPsiMass, BPHParticleMasses::jPsiMWidth);
  setParameters(Psi2, 2.0, 10.0, 3.40, 6.00, 0.0, BPHParticleMasses::psi2Mass, BPHParticleMasses::psi2MWidth);
  setParameters(Ups, 2.0, 10.0, 6.00, 12.00, 0.0, -1.0, 0.0);
  setParameters(Ups1, 2.0, 10.0, 6.00, 9.75, 0.0, BPHParticleMasses::ups1Mass, BPHParticleMasses::ups1MWidth);
  setParameters(Ups2, 2.0, 10.0, 9.75, 10.20, 0.0, BPHParticleMasses::ups2Mass, BPHParticleMasses::ups2MWidth);
  setParameters(Ups3, 2.0, 10.0, 10.20, 12.00, 0.0, BPHParticleMasses::ups3Mass, BPHParticleMasses::ups3MWidth);
  outdated = true;
}

//--------------
// Destructor --
//--------------
BPHOniaToMuMuBuilder::~BPHOniaToMuMuBuilder() {
  map<oniaType, OniaParameters>::iterator iter = oniaPar.begin();
  map<oniaType, OniaParameters>::iterator iend = oniaPar.end();
  while (iter != iend) {
    OniaParameters& par = iter++->second;
    delete par.ptSel;
    delete par.etaSel;
    delete par.massSel;
    delete par.chi2Sel;
  }
}

//--------------
// Operations --
//--------------
void BPHOniaToMuMuBuilder::fillRecList() {
  double ptMin = 9.99e+30;
  double etaMax = -1.0;
  BPHMultiSelect<BPHSlimSelect<BPHMomentumSelect>> mSel(BPHSelectOperation::or_mode);
  BPHMultiSelect<BPHSlimSelect<BPHVertexSelect>> vSel(BPHSelectOperation::or_mode);

  map<oniaType, OniaParameters>::iterator iter = oniaPar.begin();
  map<oniaType, OniaParameters>::iterator iend = oniaPar.end();
  while (iter != iend) {
    OniaParameters& par = iter++->second;
    double ptCur = par.ptSel->getPtMin();
    double etaCur = par.etaSel->getEtaMax();
    if (ptCur < ptMin)
      ptMin = ptCur;
    if (etaCur > etaMax)
      etaMax = etaCur;
    mSel.include(*par.massSel);
    vSel.include(*par.chi2Sel);
  }
  BPHMuonPtSelect ptSel(ptMin);
  BPHMuonEtaSelect etaSel(etaMax);

  BPHRecoBuilder bOnia(*evSetup);
  bOnia.add(muPosName, posCollection, BPHParticleMasses::muonMass, BPHParticleMasses::muonMSigma);
  bOnia.add(muNegName, negCollection, BPHParticleMasses::muonMass, BPHParticleMasses::muonMSigma);
  bOnia.filter(muPosName, ptSel);
  bOnia.filter(muNegName, ptSel);
  bOnia.filter(muPosName, etaSel);
  bOnia.filter(muNegName, etaSel);
  bOnia.filter(mSel);
  bOnia.filter(vSel);

  recList = BPHPlusMinusCandidate::build(bOnia, muPosName, muNegName);

  decltype(recList) tmpList;
  tmpList.reserve(recList.size());
  for (auto& c : recList) {
    auto p = c->originalReco(c->getDaug(muPosName));
    auto n = c->originalReco(c->getDaug(muNegName));
    bool accept = false;
    for (auto& e : oniaPar) {
      if (e.first == NRes)
        continue;
      auto& s = e.second;
      if ((s.ptSel->accept(*p)) && (s.ptSel->accept(*n)) && (s.etaSel->accept(*p)) && (s.etaSel->accept(*n)) &&
          (s.massSel->accept(*c)) && (s.chi2Sel->accept(*c))) {
        accept = true;
        break;
      }
    }
    if (accept)
      tmpList.push_back(c);
  }
  recList = tmpList;

  return;
}

vector<BPHPlusMinusConstCandPtr> BPHOniaToMuMuBuilder::getList(
    oniaType type, BPHRecoSelect* dSel, BPHMomentumSelect* mSel, BPHVertexSelect* vSel, BPHFitSelect* kSel) {
  extractList(type);
  vector<BPHPlusMinusConstCandPtr>& list = oniaList[type];
  int i;
  int n = list.size();
  vector<BPHPlusMinusConstCandPtr> lsub;
  lsub.reserve(n);
  for (i = 0; i < n; ++i) {
    BPHPlusMinusConstCandPtr ptr = list[i];
    const reco::Candidate* muPos = ptr->originalReco(ptr->getDaug(muPosName));
    const reco::Candidate* muNeg = ptr->originalReco(ptr->getDaug(muNegName));
    if ((dSel != nullptr) && (!dSel->accept(*muPos)))
      continue;
    if ((dSel != nullptr) && (!dSel->accept(*muNeg)))
      continue;
    if ((mSel != nullptr) && (!mSel->accept(*ptr)))
      continue;
    if ((vSel != nullptr) && (!vSel->accept(*ptr)))
      continue;
    if ((kSel != nullptr) && (!kSel->accept(*ptr)))
      continue;
    lsub.push_back(list[i]);
  }
  return lsub;
}

BPHPlusMinusConstCandPtr BPHOniaToMuMuBuilder::getOriginalCandidate(const BPHRecoCandidate& cand) {
  const reco::Candidate* mp = cand.originalReco(cand.getDaug(muPosName));
  const reco::Candidate* mn = cand.originalReco(cand.getDaug(muNegName));
  int nc = recList.size();
  int ic;
  for (ic = 0; ic < nc; ++ic) {
    BPHPlusMinusConstCandPtr pmp = recList[ic];
    const BPHPlusMinusCandidate* pmc = pmp.get();
    if (pmc->originalReco(pmc->getDaug(muPosName)) != mp)
      continue;
    if (pmc->originalReco(pmc->getDaug(muNegName)) != mn)
      continue;
    return pmp;
  }
  return BPHPlusMinusConstCandPtr(nullptr);
}

/// set cuts
void BPHOniaToMuMuBuilder::setPtMin(oniaType type, double pt) {
  setNotUpdated();
  OniaParameters& par = oniaPar[type];
  par.ptSel->setPtMin(pt);
  return;
}

void BPHOniaToMuMuBuilder::setEtaMax(oniaType type, double eta) {
  setNotUpdated();
  OniaParameters& par = oniaPar[type];
  par.etaSel->setEtaMax(eta);
  return;
}

void BPHOniaToMuMuBuilder::setMassMin(oniaType type, double m) {
  setNotUpdated();
  OniaParameters& par = oniaPar[type];
  par.massSel->setMassMin(m);
  return;
}

void BPHOniaToMuMuBuilder::setMassMax(oniaType type, double m) {
  setNotUpdated();
  OniaParameters& par = oniaPar[type];
  par.massSel->setMassMax(m);
  return;
}

void BPHOniaToMuMuBuilder::setProbMin(oniaType type, double p) {
  setNotUpdated();
  OniaParameters& par = oniaPar[type];
  par.chi2Sel->setProbMin(p);
  return;
}

void BPHOniaToMuMuBuilder::setConstr(oniaType type, double mass, double sigma) {
  setNotUpdated();
  OniaParameters& par = oniaPar[type];
  par.mass = mass;
  par.sigma = sigma;
  return;
}

/// get current cuts
double BPHOniaToMuMuBuilder::getPtMin(oniaType type) const {
  const OniaParameters& par = oniaPar.at(type);
  return par.ptSel->getPtMin();
}

double BPHOniaToMuMuBuilder::getEtaMax(oniaType type) const {
  const OniaParameters& par = oniaPar.at(type);
  return par.etaSel->getEtaMax();
}

double BPHOniaToMuMuBuilder::getMassMin(oniaType type) const {
  const OniaParameters& par = oniaPar.at(type);
  return par.massSel->getMassMin();
}

double BPHOniaToMuMuBuilder::getMassMax(oniaType type) const {
  const OniaParameters& par = oniaPar.at(type);
  return par.massSel->getMassMax();
}

double BPHOniaToMuMuBuilder::getProbMin(oniaType type) const {
  const OniaParameters& par = oniaPar.at(type);
  return par.chi2Sel->getProbMin();
}

double BPHOniaToMuMuBuilder::getConstrMass(oniaType type) const {
  const OniaParameters& par = oniaPar.at(type);
  return par.mass;
}

double BPHOniaToMuMuBuilder::getConstrSigma(oniaType type) const {
  const OniaParameters& par = oniaPar.at(type);
  return par.sigma;
}

void BPHOniaToMuMuBuilder::setNotUpdated() {
  map<oniaType, OniaParameters>::iterator iter = oniaPar.begin();
  map<oniaType, OniaParameters>::iterator iend = oniaPar.end();
  while (iter != iend)
    iter++->second.outdated = true;
  return;
}

void BPHOniaToMuMuBuilder::setParameters(oniaType type,
                                         double ptMin,
                                         double etaMax,
                                         double massMin,
                                         double massMax,
                                         double probMin,
                                         double mass,
                                         double sigma) {
  OniaParameters& par = oniaPar[type];
  par.ptSel = new BPHMuonPtSelect(ptMin);
  par.etaSel = new BPHMuonEtaSelect(etaMax);
  par.massSel = new BPHMassSelect(massMin, massMax);
  par.chi2Sel = new BPHChi2Select(probMin);
  par.mass = mass;
  par.sigma = sigma;
  par.outdated = true;
  return;
}

void BPHOniaToMuMuBuilder::extractList(oniaType type) {
  build();
  OniaParameters& par = oniaPar[type];
  vector<BPHPlusMinusConstCandPtr>& list = oniaList[type];
  if (!par.outdated)
    return;
  int i;
  int n = recList.size();
  list.clear();
  list.reserve(n);
  for (i = 0; i < n; ++i) {
    BPHPlusMinusConstCandPtr ptr = recList[i];
    const reco::Candidate* mcPos = ptr->getDaug("MuPos");
    const reco::Candidate* mcNeg = ptr->getDaug("MuNeg");
    const reco::Candidate* muPos = ptr->originalReco(mcPos);
    const reco::Candidate* muNeg = ptr->originalReco(mcNeg);
    if (!par.massSel->accept(*ptr))
      continue;
    if (!par.ptSel->accept(*muPos))
      continue;
    if (!par.etaSel->accept(*muPos))
      continue;
    if (!par.ptSel->accept(*muNeg))
      continue;
    if (!par.etaSel->accept(*muNeg))
      continue;
    if (!par.chi2Sel->accept(*ptr))
      continue;
    BPHPlusMinusCandidate* np = new BPHPlusMinusCandidate(evSetup);
    np->add("MuPos", muPos, ptr->getTrackSearchList(mcPos), BPHParticleMasses::muonMass, BPHParticleMasses::muonMSigma);
    np->add("MuNeg", muNeg, ptr->getTrackSearchList(mcNeg), BPHParticleMasses::muonMass, BPHParticleMasses::muonMSigma);
    if (par.mass > 0.0)
      np->setConstraint(par.mass, par.sigma);
    list.push_back(BPHPlusMinusConstCandPtr(np));
  }
  par.outdated = false;
  return;
}
