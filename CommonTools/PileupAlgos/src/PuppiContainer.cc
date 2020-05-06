#include "CommonTools/PileupAlgos/interface/PuppiContainer.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "Math/ProbFunc.h"
#include "TMath.h"
#include <iostream>
#include <cmath>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"

using namespace std;

PuppiContainer::PuppiContainer(const edm::ParameterSet &iConfig) {
  fPuppiDiagnostics = iConfig.getParameter<bool>("puppiDiagnostics");
  fApplyCHS = iConfig.getParameter<bool>("applyCHS");
  fInvert = iConfig.getParameter<bool>("invertPuppi");
  fUseExp = iConfig.getParameter<bool>("useExp");
  fPuppiWeightCut = iConfig.getParameter<double>("MinPuppiWeight");
  fPtMaxPhotons = iConfig.getParameter<double>("PtMaxPhotons");
  fEtaMaxPhotons = iConfig.getParameter<double>("EtaMaxPhotons");
  fPtMaxNeutrals = iConfig.getParameter<double>("PtMaxNeutrals");
  fPtMaxNeutralsStartSlope = iConfig.getParameter<double>("PtMaxNeutralsStartSlope");
  std::vector<edm::ParameterSet> lAlgos = iConfig.getParameter<std::vector<edm::ParameterSet> >("algos");
  fNAlgos = lAlgos.size();
  for (unsigned int i0 = 0; i0 < lAlgos.size(); i0++) {
    PuppiAlgo pPuppiConfig(lAlgos[i0]);
    fPuppiAlgo.push_back(pPuppiConfig);
  }
}

void PuppiContainer::initialize(const std::vector<RecoObj> &iRecoObjects) {
  //Clear everything
  fPFParticles.resize(0);
  fChargedPV.resize(0);
  fWeights.resize(0);
  fVals.resize(0);
  fRawAlphas.resize(0);
  fAlphaMed.resize(0);
  fAlphaRMS.resize(0);
  //fChargedNoPV.resize(0);
  //Link to the RecoObjects
  fPVFrac = 0.;
  fNPV = 1.;
  fRecoParticles = &iRecoObjects;
  for (auto const &rParticle : *fRecoParticles) {
    PuppiCandidate curPseudoJet;
    // float nom = sqrt((rParticle.m)*(rParticle.m) + (rParticle.pt)*(rParticle.pt)*(cosh(rParticle.eta))*(cosh(rParticle.eta))) + (rParticle.pt)*sinh(rParticle.eta);//hacked
    // float denom = sqrt((rParticle.m)*(rParticle.m) + (rParticle.pt)*(rParticle.pt));//hacked
    // float rapidity = log(nom/denom);//hacked
    if (edm::isFinite(rParticle.rapidity)) {
      curPseudoJet.reset_PtYPhiM(rParticle.pt, rParticle.rapidity, rParticle.phi, rParticle.m);  //hacked
    } else {
      curPseudoJet.reset_PtYPhiM(0, 99., 0, 0);  //skipping may have been a better choice
    }
    //curPseudoJet.reset_PtYPhiM(rParticle.pt,rParticle.eta,rParticle.phi,rParticle.m);
    // fill puppi_register
    curPseudoJet.set_info(rParticle.id);
    // fill vector of pseudojets for internal references
    fPFParticles.push_back(curPseudoJet);
    //Take Charged particles associated to PV
    if (std::abs(rParticle.id) == 1)
      fChargedPV.push_back(curPseudoJet);
    //if(rParticle.id == 3) _chargedNoPV.push_back(curPseudoJet);
    // if(fNPV < rParticle.vtxId) fNPV = rParticle.vtxId;
  }
}
PuppiContainer::~PuppiContainer() {}

double PuppiContainer::goodVar(PuppiCandidate const &iPart,
                               std::vector<PuppiCandidate> const &iParts,
                               int iOpt,
                               const double iRCone) {
  return var_within_R(iOpt, iParts, iPart, iRCone);
}

double PuppiContainer::var_within_R(int iId,
                                    const vector<PuppiCandidate> &particles,
                                    const PuppiCandidate &centre,
                                    const double R) {
  if (iId == -1)
    return 1;

  //this is a circle in rapidity-phi
  //it would make more sense to have var definition consistent
  //fastjet::Selector sel = fastjet::SelectorCircle(R);
  //sel.set_reference(centre);
  //the original code used Selector infrastructure: it is too heavy here
  //logic of SelectorCircle is preserved below

  vector<double> near_dR2s;
  near_dR2s.reserve(std::min(50UL, particles.size()));
  vector<double> near_pts;
  near_pts.reserve(std::min(50UL, particles.size()));
  const double r2 = R * R;
  for (auto const &part : particles) {
    //squared_distance is in (y,phi) coords: rap() has faster access -> check it first
    if (std::abs(part.rap() - centre.rap()) < R && part.squared_distance(centre) < r2) {
      near_dR2s.push_back(reco::deltaR2(part, centre));
      near_pts.push_back(part.pt());
    }
  }
  double var = 0;
  //double lSumPt = 0;
  //if(iId == 1) for(auto  pt : near_pts) lSumPt += pt;
  auto nParts = near_dR2s.size();
  for (auto i = 0UL; i < nParts; ++i) {
    auto dr2 = near_dR2s[i];
    auto pt = near_pts[i];
    if (dr2 < 0.0001)
      continue;
    if (iId == 0)
      var += (pt / dr2);
    else if (iId == 1)
      var += pt;
    else if (iId == 2)
      var += (1. / dr2);
    else if (iId == 3)
      var += (1. / dr2);
    else if (iId == 4)
      var += pt;
    else if (iId == 5)
      var += (pt * pt / dr2);
  }
  if (iId == 1)
    var += centre.pt();  //Sum in a cone
  else if (iId == 0 && var != 0)
    var = log(var);
  else if (iId == 3 && var != 0)
    var = log(var);
  else if (iId == 5 && var != 0)
    var = log(var);
  return var;
}
//In fact takes the median not the average
void PuppiContainer::getRMSAvg(int iOpt,
                               std::vector<PuppiCandidate> const &iConstits,
                               std::vector<PuppiCandidate> const &iParticles,
                               std::vector<PuppiCandidate> const &iChargedParticles) {
  for (unsigned int i0 = 0; i0 < iConstits.size(); i0++) {
    double pVal = -1;
    //Calculate the Puppi Algo to use
    int pPupId = getPuppiId(iConstits[i0].pt(), iConstits[i0].eta());
    if (pPupId == -1 || fPuppiAlgo[pPupId].numAlgos() <= iOpt) {
      fVals.push_back(-1);
      continue;
    }
    //Get the Puppi Sub Algo (given iteration)
    int pAlgo = fPuppiAlgo[pPupId].algoId(iOpt);
    bool pCharged = fPuppiAlgo[pPupId].isCharged(iOpt);
    double pCone = fPuppiAlgo[pPupId].coneSize(iOpt);
    //Compute the Puppi Metric
    if (!pCharged)
      pVal = goodVar(iConstits[i0], iParticles, pAlgo, pCone);
    if (pCharged)
      pVal = goodVar(iConstits[i0], iChargedParticles, pAlgo, pCone);
    fVals.push_back(pVal);
    //if(std::isnan(pVal) || std::isinf(pVal)) cerr << "====> Value is Nan " << pVal << " == " << iConstits[i0].pt() << " -- " << iConstits[i0].eta() << endl;
    if (!edm::isFinite(pVal)) {
      LogDebug("NotFound") << "====> Value is Nan " << pVal << " == " << iConstits[i0].pt() << " -- "
                           << iConstits[i0].eta() << endl;
      continue;
    }

    // // fPuppiAlgo[pPupId].add(iConstits[i0],pVal,iOpt);
    //code added by Nhan, now instead for every algorithm give it all the particles
    for (int i1 = 0; i1 < fNAlgos; i1++) {
      pAlgo = fPuppiAlgo[i1].algoId(iOpt);
      pCharged = fPuppiAlgo[i1].isCharged(iOpt);
      pCone = fPuppiAlgo[i1].coneSize(iOpt);
      double curVal = -1;
      if (i1 != pPupId) {
        if (!pCharged)
          curVal = goodVar(iConstits[i0], iParticles, pAlgo, pCone);
        if (pCharged)
          curVal = goodVar(iConstits[i0], iChargedParticles, pAlgo, pCone);
      } else {  //no need to repeat the computation
        curVal = pVal;
      }
      //std::cout << "i1 = " << i1 << ", curVal = " << curVal << ", eta = " << iConstits[i0].eta() << ", pupID = " << pPupId << std::endl;
      fPuppiAlgo[i1].add(iConstits[i0], curVal, iOpt);
    }
  }
  for (int i0 = 0; i0 < fNAlgos; i0++)
    fPuppiAlgo[i0].computeMedRMS(iOpt);
}
//In fact takes the median not the average
void PuppiContainer::getRawAlphas(int iOpt,
                                  std::vector<PuppiCandidate> const &iConstits,
                                  std::vector<PuppiCandidate> const &iParticles,
                                  std::vector<PuppiCandidate> const &iChargedParticles) {
  for (int j0 = 0; j0 < fNAlgos; j0++) {
    for (unsigned int i0 = 0; i0 < iConstits.size(); i0++) {
      double pVal = -1;
      //Get the Puppi Sub Algo (given iteration)
      int pAlgo = fPuppiAlgo[j0].algoId(iOpt);
      bool pCharged = fPuppiAlgo[j0].isCharged(iOpt);
      double pCone = fPuppiAlgo[j0].coneSize(iOpt);
      //Compute the Puppi Metric
      if (!pCharged)
        pVal = goodVar(iConstits[i0], iParticles, pAlgo, pCone);
      if (pCharged)
        pVal = goodVar(iConstits[i0], iChargedParticles, pAlgo, pCone);
      fRawAlphas.push_back(pVal);
      if (!edm::isFinite(pVal)) {
        LogDebug("NotFound") << "====> Value is Nan " << pVal << " == " << iConstits[i0].pt() << " -- "
                             << iConstits[i0].eta() << endl;
        continue;
      }
    }
  }
}
int PuppiContainer::getPuppiId(float iPt, float iEta) {
  int lId = -1;
  for (int i0 = 0; i0 < fNAlgos; i0++) {
    int nEtaBinsPerAlgo = fPuppiAlgo[i0].etaBins();
    for (int i1 = 0; i1 < nEtaBinsPerAlgo; i1++) {
      if ((std::abs(iEta) > fPuppiAlgo[i0].etaMin(i1)) && (std::abs(iEta) < fPuppiAlgo[i0].etaMax(i1))) {
        fPuppiAlgo[i0].fixAlgoEtaBin(i1);
        if (iPt > fPuppiAlgo[i0].ptMin()) {
          lId = i0;
          break;
        }
      }
    }
  }
  //if(lId == -1) std::cerr << "Error : Full fiducial range is not defined " << std::endl;
  return lId;
}
double PuppiContainer::getChi2FromdZ(double iDZ) {
  //We need to obtain prob of PU + (1-Prob of LV)
  // Prob(LV) = Gaus(dZ,sigma) where sigma = 1.5mm  (its really more like 1mm)
  //double lProbLV = ROOT::Math::normal_cdf_c(std::abs(iDZ),0.2)*2.; //*2 is to do it double sided
  //Take iDZ to be corrected by sigma already
  double lProbLV = ROOT::Math::normal_cdf_c(std::abs(iDZ), 1.) * 2.;  //*2 is to do it double sided
  double lProbPU = 1 - lProbLV;
  if (lProbPU <= 0)
    lProbPU = 1e-16;  //Quick Trick to through out infs
  if (lProbPU >= 0)
    lProbPU = 1 - 1e-16;  //Ditto
  double lChi2PU = TMath::ChisquareQuantile(lProbPU, 1);
  lChi2PU *= lChi2PU;
  return lChi2PU;
}
std::vector<double> const &PuppiContainer::puppiWeights() {
  int lNParticles = fRecoParticles->size();

  fWeights.clear();
  fWeights.reserve(lNParticles);
  fVals.clear();
  fVals.reserve(lNParticles);
  for (int i0 = 0; i0 < fNAlgos; i0++)
    fPuppiAlgo[i0].reset();

  int lNMaxAlgo = 1;
  for (int i0 = 0; i0 < fNAlgos; i0++)
    lNMaxAlgo = std::max(fPuppiAlgo[i0].numAlgos(), lNMaxAlgo);
  //Run through all compute mean and RMS
  for (int i0 = 0; i0 < lNMaxAlgo; i0++) {
    getRMSAvg(i0, fPFParticles, fPFParticles, fChargedPV);
  }
  if (fPuppiDiagnostics)
    getRawAlphas(0, fPFParticles, fPFParticles, fChargedPV);

  std::vector<double> pVals;
  pVals.reserve(lNParticles);
  for (int i0 = 0; i0 < lNParticles; i0++) {
    //Refresh
    pVals.clear();
    double pWeight = 1;
    //Get the Puppi Id and if ill defined move on
    const auto &rParticle = (*fRecoParticles)[i0];
    int pPupId = getPuppiId(rParticle.pt, rParticle.eta);
    if (pPupId == -1) {
      fWeights.push_back(0);
      fAlphaMed.push_back(-10);
      fAlphaRMS.push_back(-10);
      continue;
    }

    // fill the p-values
    double pChi2 = 0;
    if (fUseExp) {
      //Compute an Experimental Puppi Weight with delta Z info (very simple example)
      pChi2 = getChi2FromdZ(rParticle.dZ);
      //Now make sure Neutrals are not set
      if ((std::abs(rParticle.pdgId) == 22) || (std::abs(rParticle.pdgId) == 130))
        pChi2 = 0;
    }
    //Fill and compute the PuppiWeight
    int lNAlgos = fPuppiAlgo[pPupId].numAlgos();
    for (int i1 = 0; i1 < lNAlgos; i1++)
      pVals.push_back(fVals[lNParticles * i1 + i0]);

    pWeight = fPuppiAlgo[pPupId].compute(pVals, pChi2);
    //Apply the CHS weights
    if (rParticle.id == 1 && fApplyCHS)
      pWeight = 1;
    if (rParticle.id == 2 && fApplyCHS)
      pWeight = 0;
    //Basic Weight Checks
    if (!edm::isFinite(pWeight)) {
      pWeight = 0.0;
      LogDebug("PuppiWeightError") << "====> Weight is nan : " << pWeight << " : pt " << rParticle.pt
                                   << " -- eta : " << rParticle.eta << " -- Value" << fVals[i0]
                                   << " -- id :  " << rParticle.id << " --  NAlgos: " << lNAlgos << std::endl;
    }
    //Basic Cuts
    if (pWeight * fPFParticles[i0].pt() < fPuppiAlgo[pPupId].neutralPt(fNPV) && rParticle.id == 0)
      pWeight = 0;  //threshold cut on the neutral Pt
    // Protect high pT photons (important for gamma to hadronic recoil balance)
    if ((fPtMaxPhotons > 0) && (rParticle.pdgId == 22) && (std::abs(fPFParticles[i0].eta()) < fEtaMaxPhotons) &&
        (fPFParticles[i0].pt() > fPtMaxPhotons))
      pWeight = 1.;
    // Protect high pT neutrals
    else if ((fPtMaxNeutrals > 0) && (rParticle.id == 0))
      pWeight =
          std::clamp((fPFParticles[i0].pt() - fPtMaxNeutralsStartSlope) / (fPtMaxNeutrals - fPtMaxNeutralsStartSlope),
                     pWeight,
                     1.);
    if (pWeight < fPuppiWeightCut)
      pWeight = 0;  //==> Elminate the low Weight stuff
    if (fInvert)
      pWeight = 1. - pWeight;
    //std::cout << "rParticle.pt = " <<  rParticle.pt << ", rParticle.charge = " << rParticle.charge << ", rParticle.id = " << rParticle.id << ", weight = " << pWeight << std::endl;

    fWeights.push_back(pWeight);
    fAlphaMed.push_back(fPuppiAlgo[pPupId].median());
    fAlphaRMS.push_back(fPuppiAlgo[pPupId].rms());
    //Now get rid of the thrown out weights for the particle collection

    // leave these lines in, in case want to move eventually to having no 1-to-1 correspondence between puppi and pf cands
    // if( std::abs(pWeight) < std::numeric_limits<double>::denorm_min() ) continue; // this line seems not to work like it's supposed to...
    // if(std::abs(pWeight) <= 0. ) continue;
  }
  return fWeights;
}
