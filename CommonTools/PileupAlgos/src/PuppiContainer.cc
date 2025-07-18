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
  fPuppiAlgo.reserve(lAlgos.size());
  for (auto const &algos : lAlgos) {
    fPuppiAlgo.emplace_back(algos);
  }
}

void PuppiContainer::initialize(const std::vector<RecoObj> &iRecoObjects,
                                std::vector<PuppiCandidate> &pfParticles,
                                std::vector<PuppiCandidate> &pfParticlesForVar,
                                std::vector<PuppiCandidate> &pfParticlesForVarChargedPV) const {
  pfParticles.reserve(iRecoObjects.size());
  pfParticlesForVar.reserve(iRecoObjects.size());
  pfParticlesForVarChargedPV.reserve(iRecoObjects.size());
  for (auto const &rParticle : iRecoObjects) {
    PuppiCandidate pCand;
    pCand.id = rParticle.id;
    if (edm::isFinite(rParticle.rapidity)) {
      pCand.pt = rParticle.pt;
      pCand.eta = rParticle.eta;
      pCand.rapidity = rParticle.rapidity;
      pCand.phi = rParticle.phi;
      pCand.m = rParticle.m;
    } else {
      pCand.pt = 0.;
      pCand.eta = 99.;
      pCand.rapidity = 99.;
      pCand.phi = 0.;
      pCand.m = 0.;
    }

    pfParticles.push_back(pCand);

    // skip candidates to be ignored in the computation
    // of PUPPI's alphas (e.g. electrons and muons if puppiNoLep=True)
    if (std::abs(rParticle.id) == 3)
      continue;

    pfParticlesForVar.push_back(pCand);
    // charged candidates assigned to LV
    if (std::abs(rParticle.id) == 1)
      pfParticlesForVarChargedPV.push_back(pCand);
  }
}

double PuppiContainer::goodVar(PuppiCandidate const &iPart,
                               std::vector<PuppiCandidate> const &iParts,
                               int iOpt,
                               const double iRCone) const {
  return var_within_R(iOpt, iParts, iPart, iRCone);
}

double PuppiContainer::var_within_R(int iId,
                                    const vector<PuppiCandidate> &particles,
                                    const PuppiCandidate &centre,
                                    const double R) const {
  if (iId == -1)
    return 1.;

  double const r2 = R * R;
  double var = 0.;

  for (auto const &cand : particles) {
    if (std::abs(cand.rapidity - centre.rapidity) < R) {
      auto const dr2y = reco::deltaR2(cand.rapidity, cand.phi, centre.rapidity, centre.phi);
      if (dr2y < r2) {
        auto const dr2 = reco::deltaR2(cand.eta, cand.phi, centre.eta, centre.phi);
        if (dr2 < 0.0001)
          continue;
        auto const pt = cand.pt;
        switch (iId) {
          case 5:
            var += (pt * pt / dr2);
            break;
          case 4:
            var += pt;
            break;
          case 3:
            var += (1. / dr2);
            break;
          case 2:
            var += (1. / dr2);
            break;
          case 1:
            var += pt;
            break;
          case 0:
            var += (pt / dr2);
            break;
        }
      }
    }
  }

  if ((var != 0.) and ((iId == 0) or (iId == 3) or (iId == 5)))
    var = log(var);
  else if (iId == 1)
    var += centre.pt;

  return var;
}

//In fact takes the median not the average
void PuppiContainer::getRMSAvg(int iOpt,
                               std::vector<PuppiCandidate> const &iConstits,
                               std::vector<PuppiCandidate> const &iParticles,
                               std::vector<PuppiCandidate> const &iChargedParticles,
                               std::vector<double> &oVals) {
  for (unsigned int i0 = 0; i0 < iConstits.size(); i0++) {
    //Calculate the Puppi Algo to use
    int pPupId = getPuppiId(iConstits[i0].pt, iConstits[i0].eta);
    if (pPupId == -1 || fPuppiAlgo[pPupId].numAlgos() <= iOpt) {
      oVals.push_back(-1);
      continue;
    }
    //Get the Puppi Sub Algo (given iteration)
    int pAlgo = fPuppiAlgo[pPupId].algoId(iOpt);
    bool pCharged = fPuppiAlgo[pPupId].isCharged(iOpt);
    double pCone = fPuppiAlgo[pPupId].coneSize(iOpt);
    // compute the Puppi metric:
    //  - calculate goodVar only for candidates that (1) will not be assigned a predefined weight (e.g 0, 1),
    //    or (2) are required for computations inside puppi-algos (see call to PuppiAlgo::add below)
    double pVal = -1;
    bool const getsDefaultWgtIfApplyCHS = iConstits[i0].id == 1 or iConstits[i0].id == 2;
    if (not((fApplyCHS and getsDefaultWgtIfApplyCHS) or iConstits[i0].id == 3) or
        (std::abs(iConstits[i0].eta) < fPuppiAlgo[pPupId].etaMaxExtrap() and getsDefaultWgtIfApplyCHS)) {
      pVal = goodVar(iConstits[i0], pCharged ? iChargedParticles : iParticles, pAlgo, pCone);
    }
    oVals.push_back(pVal);

    if (!edm::isFinite(pVal)) {
      LogDebug("NotFound") << "====> Value is Nan " << pVal << " == " << iConstits[i0].pt << " -- " << iConstits[i0].eta
                           << endl;
      continue;
    }

    // code added by Nhan: now instead for every algorithm give it all the particles
    int count = 0;
    for (auto &algo : fPuppiAlgo) {
      int index = count++;
      // skip cands outside of algo's etaMaxExtrap, as they would anyway be ignored inside PuppiAlgo::add (see end of the block)
      if (not(std::abs(iConstits[i0].eta) < algo.etaMaxExtrap() and getsDefaultWgtIfApplyCHS))
        continue;

      auto curVal = pVal;
      // recompute goodVar if algo has changed
      if (index != pPupId) {
        pAlgo = algo.algoId(iOpt);
        pCharged = algo.isCharged(iOpt);
        pCone = algo.coneSize(iOpt);
        curVal = goodVar(iConstits[i0], pCharged ? iChargedParticles : iParticles, pAlgo, pCone);
      }

      algo.add(iConstits[i0], curVal, iOpt);
    }
  }

  for (auto &algo : fPuppiAlgo)
    algo.computeMedRMS(iOpt);
}

//In fact takes the median not the average
std::vector<double> PuppiContainer::getRawAlphas(int iOpt,
                                                 std::vector<PuppiCandidate> const &iConstits,
                                                 std::vector<PuppiCandidate> const &iParticles,
                                                 std::vector<PuppiCandidate> const &iChargedParticles) const {
  std::vector<double> oRawAlphas;
  oRawAlphas.reserve(fPuppiAlgo.size() * iConstits.size());
  for (auto &algo : fPuppiAlgo) {
    for (auto const &constit : iConstits) {
      //Get the Puppi Sub Algo (given iteration)
      int pAlgo = algo.algoId(iOpt);
      bool pCharged = algo.isCharged(iOpt);
      double pCone = algo.coneSize(iOpt);
      //Compute the Puppi Metric
      double const pVal = goodVar(constit, pCharged ? iChargedParticles : iParticles, pAlgo, pCone);
      oRawAlphas.push_back(pVal);
      if (!edm::isFinite(pVal)) {
        LogDebug("NotFound") << "====> Value is Nan " << pVal << " == " << constit.pt << " -- " << constit.eta << endl;
        continue;
      }
    }
  }
  return oRawAlphas;
}

int PuppiContainer::getPuppiId(float iPt, float iEta) {
  int lId = -1;
  int count = 0;
  for (auto &algo : fPuppiAlgo) {
    int index = count++;
    int nEtaBinsPerAlgo = algo.etaBins();
    for (int i1 = 0; i1 < nEtaBinsPerAlgo; i1++) {
      if ((std::abs(iEta) >= algo.etaMin(i1)) && (std::abs(iEta) < algo.etaMax(i1))) {
        algo.fixAlgoEtaBin(i1);
        if (iPt > algo.ptMin()) {
          lId = index;
          break;
        }
      }
    }
  }
  //if(lId == -1) std::cerr << "Error : Full fiducial range is not defined " << std::endl;
  return lId;
}
double PuppiContainer::getChi2FromdZ(double iDZ) const {
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
PuppiContainer::Weights PuppiContainer::calculatePuppiWeights(const std::vector<RecoObj> &iRecoObjects,
                                                              double iPUProxy) {
  std::vector<PuppiCandidate> pfParticles;
  std::vector<PuppiCandidate> pfParticlesForVar;
  std::vector<PuppiCandidate> pfParticlesForVarChargedPV;

  initialize(iRecoObjects, pfParticles, pfParticlesForVar, pfParticlesForVarChargedPV);

  int lNParticles = iRecoObjects.size();

  Weights returnValue;
  returnValue.weights.reserve(lNParticles);
  returnValue.puppiAlphas.reserve(lNParticles);

  //guarantee all algos are rest before leaving this function
  auto doReset = [this](void *) {
    for (auto &algo : fPuppiAlgo)
      algo.reset();
  };
  std::unique_ptr<decltype(fPuppiAlgo), decltype(doReset)> guard(&fPuppiAlgo, doReset);

  int lNMaxAlgo = 1;
  for (auto &algo : fPuppiAlgo)
    lNMaxAlgo = std::max(algo.numAlgos(), lNMaxAlgo);
  //Run through all compute mean and RMS
  for (int i0 = 0; i0 < lNMaxAlgo; i0++) {
    getRMSAvg(i0, pfParticles, pfParticlesForVar, pfParticlesForVarChargedPV, returnValue.puppiAlphas);
  }
  if (fPuppiDiagnostics)
    returnValue.puppiRawAlphas = getRawAlphas(0, pfParticles, pfParticlesForVar, pfParticlesForVarChargedPV);

  std::vector<double> pVals;
  pVals.reserve(lNParticles);
  for (int i0 = 0; i0 < lNParticles; i0++) {
    //Refresh
    pVals.clear();
    //Get the Puppi Id and if ill defined move on
    const auto &rParticle = iRecoObjects[i0];
    int pPupId = getPuppiId(rParticle.pt, rParticle.eta);
    if (pPupId == -1) {
      returnValue.weights.push_back(0);
      returnValue.puppiAlphasMed.push_back(-10);
      returnValue.puppiAlphasRMS.push_back(-10);
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
      pVals.push_back(returnValue.puppiAlphas[lNParticles * i1 + i0]);

    double pWeight = fPuppiAlgo[pPupId].compute(pVals, pChi2);
    //Apply the CHS weights
    if (rParticle.id == 1 && fApplyCHS)
      pWeight = 1;
    if (rParticle.id == 2 && fApplyCHS)
      pWeight = 0;
    //Apply weight of 1 for leptons if puppiNoLep
    if (rParticle.id == 3)
      pWeight = 1;
    //Basic Weight Checks
    if (!edm::isFinite(pWeight)) {
      pWeight = 0.0;
      LogDebug("PuppiWeightError") << "====> Weight is nan : " << pWeight << " : pt " << rParticle.pt
                                   << " -- eta : " << rParticle.eta << " -- Value" << returnValue.puppiAlphas[i0]
                                   << " -- id :  " << rParticle.id << " --  NAlgos: " << lNAlgos << std::endl;
    }
    //Basic Cuts
    if (pWeight * pfParticles[i0].pt < fPuppiAlgo[pPupId].neutralPt(iPUProxy) && rParticle.id == 0)
      pWeight = 0;  //threshold cut on the neutral Pt
    // Protect high pT photons (important for gamma to hadronic recoil balance)
    if (fPtMaxPhotons > 0 && rParticle.pdgId == 22 && std::abs(pfParticles[i0].eta) < fEtaMaxPhotons &&
        pfParticles[i0].pt > fPtMaxPhotons)
      pWeight = 1.;
    // Protect high pT neutrals
    else if ((fPtMaxNeutrals > 0) && (rParticle.id == 0))
      pWeight = std::clamp(
          (pfParticles[i0].pt - fPtMaxNeutralsStartSlope) / (fPtMaxNeutrals - fPtMaxNeutralsStartSlope), pWeight, 1.);
    if (pWeight < fPuppiWeightCut)
      pWeight = 0;  //==> Elminate the low Weight stuff
    if (fInvert)
      pWeight = 1. - pWeight;
    //std::cout << "rParticle.pt = " <<  rParticle.pt << ", rParticle.charge = " << rParticle.charge << ", rParticle.id = " << rParticle.id << ", weight = " << pWeight << std::endl;

    returnValue.weights.push_back(pWeight);
    returnValue.puppiAlphasMed.push_back(fPuppiAlgo[pPupId].median());
    returnValue.puppiAlphasRMS.push_back(fPuppiAlgo[pPupId].rms());
    //Now get rid of the thrown out returnValue.weights for the particle collection

    // leave these lines in, in case want to move eventually to having no 1-to-1 correspondence between puppi and pf cands
    // if( std::abs(pWeight) < std::numeric_limits<double>::denorm_min() ) continue; // this line seems not to work like it's supposed to...
    // if(std::abs(pWeight) <= 0. ) continue;
  }
  return returnValue;
}
