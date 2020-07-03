#include "RecoTauTag/RecoTau/interface/AntiElectronIDMVA6.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <TFile.h>
#include <array>

template <class TauType, class ElectronType>
AntiElectronIDMVA6<TauType, ElectronType>::AntiElectronIDMVA6(const edm::ParameterSet& cfg)
    : isInitialized_(false),
      mva_NoEleMatch_woGwoGSF_BL_(nullptr),
      mva_NoEleMatch_wGwoGSF_BL_(nullptr),
      mva_woGwGSF_BL_(nullptr),
      mva_wGwGSF_BL_(nullptr),
      mva_NoEleMatch_woGwoGSF_EC_(nullptr),
      mva_NoEleMatch_wGwoGSF_EC_(nullptr),
      mva_woGwGSF_EC_(nullptr),
      mva_wGwGSF_EC_(nullptr) {
  loadMVAfromDB_ = cfg.exists("loadMVAfromDB") ? cfg.getParameter<bool>("loadMVAfromDB") : false;
  if (!loadMVAfromDB_) {
    if (cfg.exists("inputFileName")) {
      inputFileName_ = cfg.getParameter<edm::FileInPath>("inputFileName");
    } else
      throw cms::Exception("MVA input not defined")
          << "Requested to load tau MVA input from ROOT file but no file provided in cfg file";
  }

  mvaName_NoEleMatch_woGwoGSF_BL_ = cfg.getParameter<std::string>("mvaName_NoEleMatch_woGwoGSF_BL");
  mvaName_NoEleMatch_wGwoGSF_BL_ = cfg.getParameter<std::string>("mvaName_NoEleMatch_wGwoGSF_BL");
  mvaName_woGwGSF_BL_ = cfg.getParameter<std::string>("mvaName_woGwGSF_BL");
  mvaName_wGwGSF_BL_ = cfg.getParameter<std::string>("mvaName_wGwGSF_BL");
  mvaName_NoEleMatch_woGwoGSF_EC_ = cfg.getParameter<std::string>("mvaName_NoEleMatch_woGwoGSF_EC");
  mvaName_NoEleMatch_wGwoGSF_EC_ = cfg.getParameter<std::string>("mvaName_NoEleMatch_wGwoGSF_EC");
  mvaName_woGwGSF_EC_ = cfg.getParameter<std::string>("mvaName_woGwGSF_EC");
  mvaName_wGwGSF_EC_ = cfg.getParameter<std::string>("mvaName_wGwGSF_EC");

  usePhiAtEcalEntranceExtrapolation_ = cfg.getParameter<bool>("usePhiAtEcalEntranceExtrapolation");

  Var_NoEleMatch_woGwoGSF_Barrel_ = new float[10];
  Var_NoEleMatch_wGwoGSF_Barrel_ = new float[18];
  Var_woGwGSF_Barrel_ = new float[24];
  Var_wGwGSF_Barrel_ = new float[32];
  Var_NoEleMatch_woGwoGSF_Endcap_ = new float[9];
  Var_NoEleMatch_wGwoGSF_Endcap_ = new float[17];
  Var_woGwGSF_Endcap_ = new float[23];
  Var_wGwGSF_Endcap_ = new float[31];

  verbosity_ = 0;
}

template <class TauType, class ElectronType>
AntiElectronIDMVA6<TauType, ElectronType>::~AntiElectronIDMVA6() {
  delete[] Var_NoEleMatch_woGwoGSF_Barrel_;
  delete[] Var_NoEleMatch_wGwoGSF_Barrel_;
  delete[] Var_woGwGSF_Barrel_;
  delete[] Var_wGwGSF_Barrel_;
  delete[] Var_NoEleMatch_woGwoGSF_Endcap_;
  delete[] Var_NoEleMatch_wGwoGSF_Endcap_;
  delete[] Var_woGwGSF_Endcap_;
  delete[] Var_wGwGSF_Endcap_;

  if (!loadMVAfromDB_) {
    delete mva_NoEleMatch_woGwoGSF_BL_;
    delete mva_NoEleMatch_wGwoGSF_BL_;
    delete mva_woGwGSF_BL_;
    delete mva_wGwGSF_BL_;
    delete mva_NoEleMatch_woGwoGSF_EC_;
    delete mva_NoEleMatch_wGwoGSF_EC_;
    delete mva_woGwGSF_EC_;
    delete mva_wGwGSF_EC_;
  }

  for (std::vector<TFile*>::iterator it = inputFilesToDelete_.begin(); it != inputFilesToDelete_.end(); ++it) {
    delete (*it);
  }
}

namespace {
  const GBRForest* loadMVAfromFile(TFile* inputFile, const std::string& mvaName) {
    const GBRForest* mva = (GBRForest*)inputFile->Get(mvaName.data());
    if (!mva)
      throw cms::Exception("PFRecoTauDiscriminationAgainstElectronMVA6::loadMVA")
          << " Failed to load MVA = " << mvaName.data() << " from file "
          << " !!\n";

    return mva;
  }

  const GBRForest* loadMVAfromDB(const edm::EventSetup& es, const std::string& mvaName) {
    edm::ESHandle<GBRForest> mva;
    es.get<GBRWrapperRcd>().get(mvaName, mva);
    return mva.product();
  }
}  // namespace

template <class TauType, class ElectronType>
void AntiElectronIDMVA6<TauType, ElectronType>::beginEvent(const edm::Event& evt, const edm::EventSetup& es) {
  if (!isInitialized_) {
    if (loadMVAfromDB_) {
      mva_NoEleMatch_woGwoGSF_BL_ = loadMVAfromDB(es, mvaName_NoEleMatch_woGwoGSF_BL_);
      mva_NoEleMatch_wGwoGSF_BL_ = loadMVAfromDB(es, mvaName_NoEleMatch_wGwoGSF_BL_);
      mva_woGwGSF_BL_ = loadMVAfromDB(es, mvaName_woGwGSF_BL_);
      mva_wGwGSF_BL_ = loadMVAfromDB(es, mvaName_wGwGSF_BL_);
      mva_NoEleMatch_woGwoGSF_EC_ = loadMVAfromDB(es, mvaName_NoEleMatch_woGwoGSF_EC_);
      mva_NoEleMatch_wGwoGSF_EC_ = loadMVAfromDB(es, mvaName_NoEleMatch_wGwoGSF_EC_);
      mva_woGwGSF_EC_ = loadMVAfromDB(es, mvaName_woGwGSF_EC_);
      mva_wGwGSF_EC_ = loadMVAfromDB(es, mvaName_wGwGSF_EC_);
    } else {
      if (inputFileName_.location() == edm::FileInPath::Unknown)
        throw cms::Exception("PFRecoTauDiscriminationAgainstElectronMVA6::loadMVA")
            << " Failed to find File = " << inputFileName_ << " !!\n";
      TFile* inputFile = new TFile(inputFileName_.fullPath().data());

      mva_NoEleMatch_woGwoGSF_BL_ = loadMVAfromFile(inputFile, mvaName_NoEleMatch_woGwoGSF_BL_);
      mva_NoEleMatch_wGwoGSF_BL_ = loadMVAfromFile(inputFile, mvaName_NoEleMatch_wGwoGSF_BL_);
      mva_woGwGSF_BL_ = loadMVAfromFile(inputFile, mvaName_woGwGSF_BL_);
      mva_wGwGSF_BL_ = loadMVAfromFile(inputFile, mvaName_wGwGSF_BL_);
      mva_NoEleMatch_woGwoGSF_EC_ = loadMVAfromFile(inputFile, mvaName_NoEleMatch_woGwoGSF_EC_);
      mva_NoEleMatch_wGwoGSF_EC_ = loadMVAfromFile(inputFile, mvaName_NoEleMatch_wGwoGSF_EC_);
      mva_woGwGSF_EC_ = loadMVAfromFile(inputFile, mvaName_woGwGSF_EC_);
      mva_wGwGSF_EC_ = loadMVAfromFile(inputFile, mvaName_wGwGSF_EC_);
      inputFilesToDelete_.push_back(inputFile);
    }
    isInitialized_ = true;
  }
  positionAtECalEntrance_.beginEvent(es);
}

template <class TauType, class ElectronType>
double AntiElectronIDMVA6<TauType, ElectronType>::MVAValue(const TauVars& tauVars,
							   const TauGammaVecs& tauGammaVecs,
							   const ElecVars& elecVars) {
  TauGammaMoms tauGammaMoms;
  double sumPt = 0.;
  double dEta2 = 0.;
  double dPhi2 = 0.;
  tauGammaMoms.signalPFGammaCandsIn = tauGammaVecs.gammasPtInSigCone.size();
  for (size_t i = 0; i < tauGammaVecs.gammasPtInSigCone.size(); ++i) {
    double pt_i = tauGammaVecs.gammasPtInSigCone[i];
    double phi_i = tauGammaVecs.gammasdPhiInSigCone[i];
    if (tauGammaVecs.gammasdPhiInSigCone[i] > M_PI)
      phi_i = tauGammaVecs.gammasdPhiInSigCone[i] - 2 * M_PI;
    else if (tauGammaVecs.gammasdPhiInSigCone[i] < -M_PI)
      phi_i = tauGammaVecs.gammasdPhiInSigCone[i] + 2 * M_PI;
    double eta_i = tauGammaVecs.gammasdEtaInSigCone[i];
    sumPt += pt_i;
    dEta2 += (pt_i * eta_i * eta_i);
    dPhi2 += (pt_i * phi_i * phi_i);
  }
  
  tauGammaMoms.gammaEnFracIn = -99.;
  if (tauVars.pt > 0.) {
    tauGammaMoms.gammaEnFracIn = sumPt / tauVars.pt;
  }
  if (sumPt > 0.) {
    dEta2 /= sumPt;
    dPhi2 /= sumPt;
  }
  tauGammaMoms.gammaEtaMomIn = std::sqrt(dEta2) * std::sqrt(tauGammaMoms.gammaEnFracIn) * tauVars.pt;
  tauGammaMoms.gammaPhiMomIn = std::sqrt(dPhi2) * std::sqrt(tauGammaMoms.gammaEnFracIn) * tauVars.pt;

  sumPt = 0.;
  dEta2 = 0.;
  dPhi2 = 0.;
  tauGammaMoms.signalPFGammaCandsOut = tauGammaVecs.gammasPtOutSigCone.size();
  for (size_t i = 0; i < tauGammaVecs.gammasPtOutSigCone.size(); ++i) {
    double pt_i = tauGammaVecs.gammasPtOutSigCone[i];
    double phi_i = tauGammaVecs.gammasdPhiOutSigCone[i];
    if (tauGammaVecs.gammasdPhiOutSigCone[i] > M_PI)
      phi_i = tauGammaVecs.gammasdPhiOutSigCone[i] - 2 * M_PI;
    else if (tauGammaVecs.gammasdPhiOutSigCone[i] < -M_PI)
      phi_i = tauGammaVecs.gammasdPhiOutSigCone[i] + 2 * M_PI;
    double eta_i = tauGammaVecs.gammasdEtaOutSigCone[i];
    sumPt += pt_i;
    dEta2 += (pt_i * eta_i * eta_i);
    dPhi2 += (pt_i * phi_i * phi_i);
  }

  tauGammaMoms.gammaEnFracOut = -99.;
  if (tauVars.pt > 0.) {
    tauGammaMoms.gammaEnFracOut = sumPt / tauVars.pt;
  }
  if (sumPt > 0.) {
    dEta2 /= sumPt;
    dPhi2 /= sumPt;
  }
  tauGammaMoms.gammaEtaMomOut = std::sqrt(dEta2) * std::sqrt(tauGammaMoms.gammaEnFracOut) * tauVars.pt;
  tauGammaMoms.gammaPhiMomOut = std::sqrt(dPhi2) * std::sqrt(tauGammaMoms.gammaEnFracOut) * tauVars.pt;

  return MVAValue(tauVars, tauGammaMoms, elecVars);
}

template <class TauType, class ElectronType>
double AntiElectronIDMVA6<TauType, ElectronType>::MVAValue(const TauVars& tauVars,
							   const TauGammaMoms& tauGammaMoms,
							   const ElecVars& elecVars) {
  if (!isInitialized_) {
    throw cms::Exception("ClassNotInitialized") << " AntiElectronMVA6 not properly initialized !!\n";
  }

  double mvaValue = -99.;

  const float ECALBarrelEndcapEtaBorder = 1.479;
  float elecDeltaPinPoutOverPin = (elecVars.pIn > 0.0) ? (std::abs(elecVars.pIn - elecVars.pOut) / elecVars.pIn) : 1.0;
  float elecEecalOverPout = (elecVars.pOut > 0.0) ? (elecVars.eEcal / elecVars.pOut) : 20.0;
  float elecNumHitsDiffOverSum = ((elecVars.gsfNumHits + elecVars.kfNumHits) > 0.0)
                                     ? ((elecVars.gsfNumHits - elecVars.kfNumHits) / (elecVars.gsfNumHits + elecVars.kfNumHits))
                                     : 1.0;

  if (deltaR(tauVars.etaAtEcalEntrance, tauVars.phi, elecVars.eta, elecVars.phi) > 0.3 && tauGammaMoms.signalPFGammaCandsIn == 0 && tauVars.hasGsf < 0.5) {
    if (std::abs(tauVars.etaAtEcalEntrance) < ECALBarrelEndcapEtaBorder) {
      Var_NoEleMatch_woGwoGSF_Barrel_[0] = tauVars.etaAtEcalEntrance;
      Var_NoEleMatch_woGwoGSF_Barrel_[1] = tauVars.leadChargedPFCandEtaAtEcalEntrance;
      Var_NoEleMatch_woGwoGSF_Barrel_[2] = std::min(float(2.), tauVars.leadChargedPFCandPt / std::max(float(1.), tauVars.pt));
      Var_NoEleMatch_woGwoGSF_Barrel_[3] = std::log(std::max(float(1.), tauVars.pt));
      Var_NoEleMatch_woGwoGSF_Barrel_[4] = tauVars.emFraction;
      Var_NoEleMatch_woGwoGSF_Barrel_[5] = tauVars.leadPFChargedHadrHoP;
      Var_NoEleMatch_woGwoGSF_Barrel_[6] = tauVars.leadPFChargedHadrEoP;
      Var_NoEleMatch_woGwoGSF_Barrel_[7] = tauVars.visMassIn;
      Var_NoEleMatch_woGwoGSF_Barrel_[8] = tauVars.dCrackEta;
      Var_NoEleMatch_woGwoGSF_Barrel_[9] = tauVars.dCrackPhi;
      mvaValue = mva_NoEleMatch_woGwoGSF_BL_->GetClassifier(Var_NoEleMatch_woGwoGSF_Barrel_);
    } else {
      Var_NoEleMatch_woGwoGSF_Endcap_[0] = tauVars.etaAtEcalEntrance;
      Var_NoEleMatch_woGwoGSF_Endcap_[1] = tauVars.leadChargedPFCandEtaAtEcalEntrance;
      Var_NoEleMatch_woGwoGSF_Endcap_[2] = std::min(float(2.), tauVars.leadChargedPFCandPt / std::max(float(1.), tauVars.pt));
      Var_NoEleMatch_woGwoGSF_Endcap_[3] = std::log(std::max(float(1.), tauVars.pt));
      Var_NoEleMatch_woGwoGSF_Endcap_[4] = tauVars.emFraction;
      Var_NoEleMatch_woGwoGSF_Endcap_[5] = tauVars.leadPFChargedHadrHoP;
      Var_NoEleMatch_woGwoGSF_Endcap_[6] = tauVars.leadPFChargedHadrEoP;
      Var_NoEleMatch_woGwoGSF_Endcap_[7] = tauVars.visMassIn;
      Var_NoEleMatch_woGwoGSF_Endcap_[8] = tauVars.dCrackEta;
      mvaValue = mva_NoEleMatch_woGwoGSF_EC_->GetClassifier(Var_NoEleMatch_woGwoGSF_Endcap_);
    }
  } else if (deltaR(tauVars.etaAtEcalEntrance, tauVars.phi, elecVars.eta, elecVars.phi) > 0.3 &&
	    tauGammaMoms.signalPFGammaCandsIn > 0 && tauVars.hasGsf < 0.5) {
    if (std::abs(tauVars.etaAtEcalEntrance) < ECALBarrelEndcapEtaBorder) {
      Var_NoEleMatch_wGwoGSF_Barrel_[0] = tauVars.etaAtEcalEntrance;
      Var_NoEleMatch_wGwoGSF_Barrel_[1] = tauVars.leadChargedPFCandEtaAtEcalEntrance;
      Var_NoEleMatch_wGwoGSF_Barrel_[2] = std::min(float(2.), tauVars.leadChargedPFCandPt / std::max(float(1.), tauVars.pt));
      Var_NoEleMatch_wGwoGSF_Barrel_[3] = std::log(std::max(float(1.), tauVars.pt));
      Var_NoEleMatch_wGwoGSF_Barrel_[4] = tauVars.emFraction;
      Var_NoEleMatch_wGwoGSF_Barrel_[5] = tauGammaMoms.signalPFGammaCandsIn;
      Var_NoEleMatch_wGwoGSF_Barrel_[6] = tauGammaMoms.signalPFGammaCandsOut;
      Var_NoEleMatch_wGwoGSF_Barrel_[7] = tauVars.leadPFChargedHadrHoP;
      Var_NoEleMatch_wGwoGSF_Barrel_[8] = tauVars.leadPFChargedHadrEoP;
      Var_NoEleMatch_wGwoGSF_Barrel_[9] = tauVars.visMassIn;
      Var_NoEleMatch_wGwoGSF_Barrel_[10] = tauGammaMoms.gammaEtaMomIn;
      Var_NoEleMatch_wGwoGSF_Barrel_[11] = tauGammaMoms.gammaEtaMomOut;
      Var_NoEleMatch_wGwoGSF_Barrel_[12] = tauGammaMoms.gammaPhiMomIn;
      Var_NoEleMatch_wGwoGSF_Barrel_[13] = tauGammaMoms.gammaPhiMomOut;
      Var_NoEleMatch_wGwoGSF_Barrel_[14] = tauGammaMoms.gammaEnFracIn;
      Var_NoEleMatch_wGwoGSF_Barrel_[15] = tauGammaMoms.gammaEnFracOut;
      Var_NoEleMatch_wGwoGSF_Barrel_[16] = tauVars.dCrackEta;
      Var_NoEleMatch_wGwoGSF_Barrel_[17] = tauVars.dCrackPhi;
      mvaValue = mva_NoEleMatch_wGwoGSF_BL_->GetClassifier(Var_NoEleMatch_wGwoGSF_Barrel_);
    } else {
      Var_NoEleMatch_wGwoGSF_Endcap_[0] = tauVars.etaAtEcalEntrance;
      Var_NoEleMatch_wGwoGSF_Endcap_[1] = tauVars.leadChargedPFCandEtaAtEcalEntrance;
      Var_NoEleMatch_wGwoGSF_Endcap_[2] = std::min(float(2.), tauVars.leadChargedPFCandPt / std::max(float(1.), tauVars.pt));
      Var_NoEleMatch_wGwoGSF_Endcap_[3] = std::log(std::max(float(1.), tauVars.pt));
      Var_NoEleMatch_wGwoGSF_Endcap_[4] = tauVars.emFraction;
      Var_NoEleMatch_wGwoGSF_Endcap_[5] = tauGammaMoms.signalPFGammaCandsIn;
      Var_NoEleMatch_wGwoGSF_Endcap_[6] = tauGammaMoms.signalPFGammaCandsOut;
      Var_NoEleMatch_wGwoGSF_Endcap_[7] = tauVars.leadPFChargedHadrHoP;
      Var_NoEleMatch_wGwoGSF_Endcap_[8] = tauVars.leadPFChargedHadrEoP;
      Var_NoEleMatch_wGwoGSF_Endcap_[9] = tauVars.visMassIn;
      Var_NoEleMatch_wGwoGSF_Endcap_[10] = tauGammaMoms.gammaEtaMomIn;
      Var_NoEleMatch_wGwoGSF_Endcap_[11] = tauGammaMoms.gammaEtaMomOut;
      Var_NoEleMatch_wGwoGSF_Endcap_[12] = tauGammaMoms.gammaPhiMomIn;
      Var_NoEleMatch_wGwoGSF_Endcap_[13] = tauGammaMoms.gammaPhiMomOut;
      Var_NoEleMatch_wGwoGSF_Endcap_[14] = tauGammaMoms.gammaEnFracIn;
      Var_NoEleMatch_wGwoGSF_Endcap_[15] = tauGammaMoms.gammaEnFracOut;
      Var_NoEleMatch_wGwoGSF_Endcap_[16] = tauVars.dCrackEta;
      mvaValue = mva_NoEleMatch_wGwoGSF_EC_->GetClassifier(Var_NoEleMatch_wGwoGSF_Endcap_);
    }
  } else if (tauGammaMoms.signalPFGammaCandsIn == 0 && tauVars.hasGsf > 0.5) {
    if (std::abs(tauVars.etaAtEcalEntrance) < ECALBarrelEndcapEtaBorder) {
      Var_woGwGSF_Barrel_[0] = std::max(float(-0.1), elecVars.eTotOverPin);
      Var_woGwGSF_Barrel_[1] = std::log(std::max(float(0.01), elecVars.chi2NormGSF));
      Var_woGwGSF_Barrel_[2] = elecVars.gsfNumHits;
      Var_woGwGSF_Barrel_[3] = std::log(std::max(float(0.01), elecVars.gsfTrackResol));
      Var_woGwGSF_Barrel_[4] = elecVars.gsfTracklnPt;
      Var_woGwGSF_Barrel_[5] = elecNumHitsDiffOverSum;
      Var_woGwGSF_Barrel_[6] = std::log(std::max(float(0.01), elecVars.chi2NormKF));
      Var_woGwGSF_Barrel_[7] = std::min(elecDeltaPinPoutOverPin, float(1.));
      Var_woGwGSF_Barrel_[8] = std::min(elecEecalOverPout, float(20.));
      Var_woGwGSF_Barrel_[9] = elecVars.deltaEta;
      Var_woGwGSF_Barrel_[10] = elecVars.deltaPhi;
      Var_woGwGSF_Barrel_[11] = std::min(elecVars.mvaInSigmaEtaEta, float(0.01));
      Var_woGwGSF_Barrel_[12] = std::min(elecVars.mvaInHadEnergy, float(20.));
      Var_woGwGSF_Barrel_[13] = std::min(elecVars.mvaInDeltaEta, float(0.1));
      Var_woGwGSF_Barrel_[14] = tauVars.etaAtEcalEntrance;
      Var_woGwGSF_Barrel_[15] = tauVars.leadChargedPFCandEtaAtEcalEntrance;
      Var_woGwGSF_Barrel_[16] = std::min(float(2.), tauVars.leadChargedPFCandPt / std::max(float(1.), tauVars.pt));
      Var_woGwGSF_Barrel_[17] = std::log(std::max(float(1.), tauVars.pt));
      Var_woGwGSF_Barrel_[18] = tauVars.emFraction;
      Var_woGwGSF_Barrel_[19] = tauVars.leadPFChargedHadrHoP;
      Var_woGwGSF_Barrel_[20] = tauVars.leadPFChargedHadrEoP;
      Var_woGwGSF_Barrel_[21] = tauVars.visMassIn;
      Var_woGwGSF_Barrel_[22] = tauVars.dCrackEta;
      Var_woGwGSF_Barrel_[23] = tauVars.dCrackPhi;
      mvaValue = mva_woGwGSF_BL_->GetClassifier(Var_woGwGSF_Barrel_);
    } else {
      Var_woGwGSF_Endcap_[0] = std::max(float(-0.1), elecVars.eTotOverPin);
      Var_woGwGSF_Endcap_[1] = std::log(std::max(float(0.01), elecVars.chi2NormGSF));
      Var_woGwGSF_Endcap_[2] = elecVars.gsfNumHits;
      Var_woGwGSF_Endcap_[3] = std::log(std::max(float(0.01), elecVars.gsfTrackResol));
      Var_woGwGSF_Endcap_[4] = elecVars.gsfTracklnPt;
      Var_woGwGSF_Endcap_[5] = elecNumHitsDiffOverSum;
      Var_woGwGSF_Endcap_[6] = std::log(std::max(float(0.01), elecVars.chi2NormKF));
      Var_woGwGSF_Endcap_[7] = std::min(elecDeltaPinPoutOverPin, float(1.));
      Var_woGwGSF_Endcap_[8] = std::min(elecEecalOverPout, float(20.));
      Var_woGwGSF_Endcap_[9] = elecVars.deltaEta;
      Var_woGwGSF_Endcap_[10] = elecVars.deltaPhi;
      Var_woGwGSF_Endcap_[11] = std::min(elecVars.mvaInSigmaEtaEta, float(0.01));
      Var_woGwGSF_Endcap_[12] = std::min(elecVars.mvaInHadEnergy, float(20.));
      Var_woGwGSF_Endcap_[13] = std::min(elecVars.mvaInDeltaEta, float(0.1));
      Var_woGwGSF_Endcap_[14] = tauVars.etaAtEcalEntrance;
      Var_woGwGSF_Endcap_[15] = tauVars.leadChargedPFCandEtaAtEcalEntrance;
      Var_woGwGSF_Endcap_[16] = std::min(float(2.), tauVars.leadChargedPFCandPt / std::max(float(1.), tauVars.pt));
      Var_woGwGSF_Endcap_[17] = std::log(std::max(float(1.), tauVars.pt));
      Var_woGwGSF_Endcap_[18] = tauVars.emFraction;
      Var_woGwGSF_Endcap_[19] = tauVars.leadPFChargedHadrHoP;
      Var_woGwGSF_Endcap_[20] = tauVars.leadPFChargedHadrEoP;
      Var_woGwGSF_Endcap_[21] = tauVars.visMassIn;
      Var_woGwGSF_Endcap_[22] = tauVars.dCrackEta;
      mvaValue = mva_woGwGSF_EC_->GetClassifier(Var_woGwGSF_Endcap_);
    }
  } else if (tauGammaMoms.signalPFGammaCandsIn > 0 && tauVars.hasGsf > 0.5) {
    if (std::abs(tauVars.etaAtEcalEntrance) < ECALBarrelEndcapEtaBorder) {
      Var_wGwGSF_Barrel_[0] = std::max(float(-0.1), elecVars.eTotOverPin);
      Var_wGwGSF_Barrel_[1] = std::log(std::max(float(0.01), elecVars.chi2NormGSF));
      Var_wGwGSF_Barrel_[2] = elecVars.gsfNumHits;
      Var_wGwGSF_Barrel_[3] = std::log(std::max(float(0.01), elecVars.gsfTrackResol));
      Var_wGwGSF_Barrel_[4] = elecVars.gsfTracklnPt;
      Var_wGwGSF_Barrel_[5] = elecNumHitsDiffOverSum;
      Var_wGwGSF_Barrel_[6] = std::log(std::max(float(0.01), elecVars.chi2NormKF));
      Var_wGwGSF_Barrel_[7] = std::min(elecDeltaPinPoutOverPin, float(1.));
      Var_wGwGSF_Barrel_[8] = std::min(elecEecalOverPout, float(20.));
      Var_wGwGSF_Barrel_[9] = elecVars.deltaEta;
      Var_wGwGSF_Barrel_[10] = elecVars.deltaPhi;
      Var_wGwGSF_Barrel_[11] = std::min(elecVars.mvaInSigmaEtaEta, float(0.01));
      Var_wGwGSF_Barrel_[12] = std::min(elecVars.mvaInHadEnergy, float(20.));
      Var_wGwGSF_Barrel_[13] = std::min(elecVars.mvaInDeltaEta, float(0.1));
      Var_wGwGSF_Barrel_[14] = tauVars.etaAtEcalEntrance;
      Var_wGwGSF_Barrel_[15] = tauVars.leadChargedPFCandEtaAtEcalEntrance;
      Var_wGwGSF_Barrel_[16] = std::min(float(2.), tauVars.leadChargedPFCandPt / std::max(float(1.), tauVars.pt));
      Var_wGwGSF_Barrel_[17] = std::log(std::max(float(1.), tauVars.pt));
      Var_wGwGSF_Barrel_[18] = tauVars.emFraction;
      Var_wGwGSF_Barrel_[19] = tauGammaMoms.signalPFGammaCandsIn;
      Var_wGwGSF_Barrel_[20] = tauGammaMoms.signalPFGammaCandsOut;
      Var_wGwGSF_Barrel_[21] = tauVars.leadPFChargedHadrHoP;
      Var_wGwGSF_Barrel_[22] = tauVars.leadPFChargedHadrEoP;
      Var_wGwGSF_Barrel_[23] = tauVars.visMassIn;
      Var_wGwGSF_Barrel_[24] = tauGammaMoms.gammaEtaMomIn;
      Var_wGwGSF_Barrel_[25] = tauGammaMoms.gammaEtaMomOut;
      Var_wGwGSF_Barrel_[26] = tauGammaMoms.gammaPhiMomIn;
      Var_wGwGSF_Barrel_[27] = tauGammaMoms.gammaPhiMomOut;
      Var_wGwGSF_Barrel_[28] = tauGammaMoms.gammaEnFracIn;
      Var_wGwGSF_Barrel_[29] = tauGammaMoms.gammaEnFracOut;
      Var_wGwGSF_Barrel_[30] = tauVars.dCrackEta;
      Var_wGwGSF_Barrel_[31] = tauVars.dCrackPhi;
      mvaValue = mva_wGwGSF_BL_->GetClassifier(Var_wGwGSF_Barrel_);
    } else {
      Var_wGwGSF_Endcap_[0] = std::max(float(-0.1), elecVars.eTotOverPin);
      Var_wGwGSF_Endcap_[1] = std::log(std::max(float(0.01), elecVars.chi2NormGSF));
      Var_wGwGSF_Endcap_[2] = elecVars.gsfNumHits;
      Var_wGwGSF_Endcap_[3] = std::log(std::max(float(0.01), elecVars.gsfTrackResol));
      Var_wGwGSF_Endcap_[4] = elecVars.gsfTracklnPt;
      Var_wGwGSF_Endcap_[5] = elecNumHitsDiffOverSum;
      Var_wGwGSF_Endcap_[6] = std::log(std::max(float(0.01), elecVars.chi2NormKF));
      Var_wGwGSF_Endcap_[7] = std::min(elecDeltaPinPoutOverPin, float(1.));
      Var_wGwGSF_Endcap_[8] = std::min(elecEecalOverPout, float(20.));
      Var_wGwGSF_Endcap_[9] = elecVars.deltaEta;
      Var_wGwGSF_Endcap_[10] = elecVars.deltaPhi;
      Var_wGwGSF_Endcap_[11] = std::min(elecVars.mvaInSigmaEtaEta, float(0.01));
      Var_wGwGSF_Endcap_[12] = std::min(elecVars.mvaInHadEnergy, float(20.));
      Var_wGwGSF_Endcap_[13] = std::min(elecVars.mvaInDeltaEta, float(0.1));
      Var_wGwGSF_Endcap_[14] = tauVars.etaAtEcalEntrance;
      Var_wGwGSF_Endcap_[15] = tauVars.leadChargedPFCandEtaAtEcalEntrance;
      Var_wGwGSF_Endcap_[16] = std::min(float(2.), tauVars.leadChargedPFCandPt / std::max(float(1.), tauVars.pt));
      Var_wGwGSF_Endcap_[17] = std::log(std::max(float(1.), tauVars.pt));
      Var_wGwGSF_Endcap_[18] = tauVars.emFraction;
      Var_wGwGSF_Endcap_[19] = tauGammaMoms.signalPFGammaCandsIn;
      Var_wGwGSF_Endcap_[20] = tauGammaMoms.signalPFGammaCandsOut;
      Var_wGwGSF_Endcap_[21] = tauVars.leadPFChargedHadrHoP;
      Var_wGwGSF_Endcap_[22] = tauVars.leadPFChargedHadrEoP;
      Var_wGwGSF_Endcap_[23] = tauVars.visMassIn;
      Var_wGwGSF_Endcap_[24] = tauGammaMoms.gammaEtaMomIn;
      Var_wGwGSF_Endcap_[25] = tauGammaMoms.gammaEtaMomOut;
      Var_wGwGSF_Endcap_[26] = tauGammaMoms.gammaPhiMomIn;
      Var_wGwGSF_Endcap_[27] = tauGammaMoms.gammaPhiMomOut;
      Var_wGwGSF_Endcap_[28] = tauGammaMoms.gammaEnFracIn;
      Var_wGwGSF_Endcap_[29] = tauGammaMoms.gammaEnFracOut;
      Var_wGwGSF_Endcap_[30] = tauVars.dCrackEta;
      mvaValue = mva_wGwGSF_EC_->GetClassifier(Var_wGwGSF_Endcap_);
    }
  }
  return mvaValue;
}

template <class TauType, class ElectronType>
double AntiElectronIDMVA6<TauType, ElectronType>::MVAValue(const TauType& theTau, const ElectronType& theEle)

{
  // === tau variables ===
  TauVars tauVars = AntiElectronIDMVA6<TauType, ElectronType>::getTauVars(theTau);  
  TauGammaVecs tauGammaVecs = AntiElectronIDMVA6<TauType, ElectronType>::getTauGammaVecs(theTau);

  // === electron variables ===
  ElecVars elecVars = AntiElectronIDMVA6<TauType, ElectronType>::getElecVars(theEle);

  return MVAValue(tauVars, tauGammaVecs, elecVars);
}

template <class TauType, class ElectronType>
double AntiElectronIDMVA6<TauType, ElectronType>::MVAValue(const TauType& theTau) {
  // === tau variables ===
  TauVars tauVars = AntiElectronIDMVA6<TauType, ElectronType>::getTauVars(theTau);
  TauGammaVecs tauGammaVecs = AntiElectronIDMVA6<TauType, ElectronType>::getTauGammaVecs(theTau);

  // === electron variables ===
  ElecVars elecVars;
  elecVars.eta = 9.9; //Dummy value used in MVA training

  return MVAValue(tauVars, tauGammaVecs, elecVars);
}

template <class TauType, class ElectronType>
TauVars AntiElectronIDMVA6<TauType, ElectronType>::getTauVars(const TauType& theTau){
  TauVars tauVars;
  if (std::is_same<TauType, reco::PFTau>::value ||
      std::is_same<TauType, pat::Tau>::value)
    tauVars = getTauVarsTypeSpecific(theTau);
  else
    throw cms::Exception("AntiElectronIDMVA6")
      << "Unsupported TauType used. You must use either reco::PFTau or pat::Tau.";

  tauVars.pt = theTau.pt();

  reco::Candidate::LorentzVector pfGammaSum(0, 0, 0, 0);
  reco::Candidate::LorentzVector pfChargedSum(0, 0, 0, 0);
  float signalrad = std::clamp(3.0 / std::max(1.0, theTau.pt()), 0.05, 0.10);
  for (const auto& gamma : theTau.signalGammaCands()) {
    float dR = deltaR(gamma->p4(), theTau.leadChargedHadrCand()->p4());
    // pfGammas inside the tau signal cone
    if (dR < signalrad) {
      pfGammaSum += gamma->p4();
    }
  }
  for (const auto& charged : theTau.signalChargedHadrCands()) {
    float dR = deltaR(charged->p4(), theTau.leadChargedHadrCand()->p4());
    // charged particles inside the tau signal cone
    if (dR < signalrad) {
      pfChargedSum += charged->p4();
    }
  }
  tauVars.visMassIn = (pfGammaSum + pfChargedSum).mass();

  tauVars.hasGsf = 0;
  if (theTau.leadChargedHadrCand().isNonnull()) {
    const pat::PackedCandidate* packedLeadChCand =
      dynamic_cast<const pat::PackedCandidate*>(theTau.leadChargedHadrCand().get());
    if (packedLeadChCand != nullptr) {
      if (std::abs(packedLeadChCand->pdgId()) == 11)
        tauVars.hasGsf = 1;
    } else {
      const reco::PFCandidate* pfLeadChCand =
        dynamic_cast<const reco::PFCandidate*>(theTau.leadChargedHadrCand().get());
      if (pfLeadChCand != nullptr && pfLeadChCand->gsfTrackRef().isNonnull())
        tauVars.hasGsf = 1;
    }
  }
  tauVars.dCrackPhi = dCrackPhi(tauVars.phi, tauVars.etaAtEcalEntrance);
  tauVars.dCrackEta = dCrackEta(tauVars.etaAtEcalEntrance);

  return tauVars;
}

template <class TauType, class ElectronType>
TauGammaVecs AntiElectronIDMVA6<TauType, ElectronType>::getTauGammaVecs(const TauType& theTau){
  TauGammaVecs tauGammaVecs;

  float signalrad = std::clamp(3.0 / std::max(1.0, theTau.pt()), 0.05, 0.10);
  for (const auto& gamma : theTau.signalGammaCands()) {
    float dR = deltaR(gamma->p4(), theTau.leadChargedHadrCand()->p4());
    // pfGammas inside the tau signal cone
    if (dR < signalrad) {
      tauGammaVecs.gammasdEtaInSigCone.push_back(gamma->eta() - theTau.leadChargedHadrCand()->eta());
      tauGammaVecs.gammasdPhiInSigCone.push_back(gamma->phi() - theTau.leadChargedHadrCand()->phi());
      tauGammaVecs.gammasPtInSigCone.push_back(gamma->pt());
    }
    // pfGammas outside the tau signal cone
    else {
      tauGammaVecs.gammasdEtaOutSigCone.push_back(gamma->eta() - theTau.leadChargedHadrCand()->eta());
      tauGammaVecs.gammasdPhiOutSigCone.push_back(gamma->phi() - theTau.leadChargedHadrCand()->phi());
      tauGammaVecs.gammasPtOutSigCone.push_back(gamma->pt());
    }
  }
  return tauGammaVecs;
}

template <class TauType, class ElectronType>
ElecVars AntiElectronIDMVA6<TauType, ElectronType>::getElecVars(const ElectronType& theEle){

  ElecVars elecVars;

  elecVars.eta = theEle.eta();
  elecVars.phi = theEle.phi();

  // Variables related to the electron Cluster
  float elecEe = 0.;
  float elecEgamma = 0.;
  reco::SuperClusterRef pfSuperCluster = theEle.superCluster();
  if (pfSuperCluster.isNonnull() && pfSuperCluster.isAvailable()) {
    for (reco::CaloCluster_iterator pfCluster = pfSuperCluster->clustersBegin();
         pfCluster != pfSuperCluster->clustersEnd();
         ++pfCluster) {
      double pfClusterEn = (*pfCluster)->energy();
      if (pfCluster == pfSuperCluster->clustersBegin())
        elecEe += pfClusterEn;
      else
        elecEgamma += pfClusterEn;
    }
  }

  elecVars.pIn = std::sqrt(theEle.trackMomentumAtVtx().Mag2());
  elecVars.pOut = std::sqrt(theEle.trackMomentumOut().Mag2());
  elecVars.eTotOverPin = (elecVars.pIn > 0.0) ? ((elecEe + elecEgamma) / elecVars.pIn) : -0.1;
  elecVars.eEcal = theEle.ecalEnergy();
  elecVars.deltaEta = theEle.deltaEtaSeedClusterTrackAtCalo();
  elecVars.deltaPhi = theEle.deltaPhiSeedClusterTrackAtCalo();
  elecVars.mvaInSigmaEtaEta = theEle.mvaInput().sigmaEtaEta;
  elecVars.mvaInHadEnergy = theEle.mvaInput().hadEnergy;
  elecVars.mvaInDeltaEta = theEle.mvaInput().deltaEta;

  // Variables related to the GsfTrack
  elecVars.chi2NormGSF = -99.;
  elecVars.gsfNumHits = -99.;
  elecVars.gsfTrackResol = -99.;
  elecVars.gsfTracklnPt = -99.;
  if (theEle.gsfTrack().isNonnull()) {
    elecVars.chi2NormGSF = theEle.gsfTrack()->normalizedChi2();
    elecVars.gsfNumHits = theEle.gsfTrack()->numberOfValidHits();
    if (theEle.gsfTrack()->pt() > 0.) {
      elecVars.gsfTrackResol = theEle.gsfTrack()->ptError() / theEle.gsfTrack()->pt();
      elecVars.gsfTracklnPt = log(theEle.gsfTrack()->pt()) * M_LN10;
    }
  }

  // Variables related to the CtfTrack
  elecVars.chi2NormKF = -99.;
  elecVars.kfNumHits = -99.;
  if (theEle.closestCtfTrackRef().isNonnull()) {
    elecVars.chi2NormKF = theEle.closestCtfTrackRef()->normalizedChi2();
    elecVars.kfNumHits = theEle.closestCtfTrackRef()->numberOfValidHits();
  }

  return elecVars;
}

template <class TauType, class ElectronType>
double AntiElectronIDMVA6<TauType, ElectronType>::minimum(double a, double b) {
  if (std::abs(b) < std::abs(a))
    return b;
  else
    return a;
}

namespace {

  // IN: define locations of the 18 phi-cracks
  std::array<double, 18> fill_cPhi() {
    constexpr double pi = M_PI;  // 3.14159265358979323846;
    std::array<double, 18> cPhi;
    // IN: define locations of the 18 phi-cracks
    cPhi[0] = 2.97025;
    for (unsigned iCrack = 1; iCrack <= 17; ++iCrack) {
      cPhi[iCrack] = cPhi[0] - 2. * iCrack * pi / 18;
    }
    return cPhi;
  }

  const std::array<double, 18> cPhi = fill_cPhi();

}  // namespace

template <class TauType, class ElectronType>
double AntiElectronIDMVA6<TauType, ElectronType>::dCrackPhi(double phi, double eta) {
  //--- compute the (unsigned) distance to the closest phi-crack in the ECAL barrel

  constexpr double pi = M_PI;  // 3.14159265358979323846;

  // IN: shift of this location if eta < 0
  constexpr double delta_cPhi = 0.00638;

  double retVal = 99.;

  if (eta >= -1.47464 && eta <= 1.47464) {
    // the location is shifted
    if (eta < 0.)
      phi += delta_cPhi;

    // CV: need to bring-back phi into interval [-pi,+pi]
    if (phi > pi)
      phi -= 2. * pi;
    if (phi < -pi)
      phi += 2. * pi;

    if (phi >= -pi && phi <= pi) {
      // the problem of the extrema:
      if (phi < cPhi[17] || phi >= cPhi[0]) {
        if (phi < 0.)
          phi += 2. * pi;
        retVal = minimum(phi - cPhi[0], phi - cPhi[17] - 2. * pi);
      } else {
        // between these extrema...
        bool OK = false;
        unsigned iCrack = 16;
        while (!OK) {
          if (phi < cPhi[iCrack]) {
            retVal = minimum(phi - cPhi[iCrack + 1], phi - cPhi[iCrack]);
            OK = true;
          } else {
            iCrack -= 1;
          }
        }
      }
    } else {
      retVal = 0.;  // IN: if there is a problem, we assume that we are in a crack
    }
  } else {
    return -99.;
  }

  return std::abs(retVal);
}

template <class TauType, class ElectronType>
double AntiElectronIDMVA6<TauType, ElectronType>::dCrackEta(double eta) {
  //--- compute the (unsigned) distance to the closest eta-crack in the ECAL barrel

  // IN: define locations of the eta-cracks
  double cracks[5] = {0., 4.44747e-01, 7.92824e-01, 1.14090e+00, 1.47464e+00};

  double retVal = 99.;

  for (int iCrack = 0; iCrack < 5; ++iCrack) {
    double d = minimum(eta - cracks[iCrack], eta + cracks[iCrack]);
    if (std::abs(d) < std::abs(retVal)) {
      retVal = d;
    }
  }

  return std::abs(retVal);
}

// pat::Tau
template <class TauType, class ElectronType>
TauVars AntiElectronIDMVA6<TauType, ElectronType>::getTauVarsTypeSpecific(const pat::Tau& theTau) {
  TauVars tauVars;
  tauVars.etaAtEcalEntrance = theTau.etaAtEcalEntrance();
  tauVars.leadChargedPFCandEtaAtEcalEntrance = theTau.etaAtEcalEntranceLeadChargedCand();
  tauVars.leadChargedPFCandPt = theTau.ptLeadChargedCand();
  tauVars.phi = theTau.phi();
  if (!usePhiAtEcalEntranceExtrapolation_) {
    tauVars.phi = theTau.phiAtEcalEntrance();
  } else {
    float sumPhiTimesEnergy = 0.;
    float sumEnergy = 0.;
    for (const auto& candidate : theTau.signalCands()) {
      float phiAtECalEntrance = candidate->phi();
      bool success = false;
      reco::Candidate::Point posAtECal =
	positionAtECalEntrance_(candidate.get(), success);
      if (success) {
        phiAtECalEntrance = posAtECal.phi();
      }
      sumPhiTimesEnergy += phiAtECalEntrance * candidate->energy();
      sumEnergy += candidate->energy();
    }
    if (sumEnergy > 0.) {
      tauVars.phi = sumPhiTimesEnergy / sumEnergy;
    }
  }
  tauVars.emFraction = std::max(theTau.emFraction_MVA(), (float)0.);
  tauVars.leadPFChargedHadrHoP = 0.;
  tauVars.leadPFChargedHadrEoP = 0.;
  if (theTau.leadChargedHadrCand()->p() > 0.) {
    tauVars.leadPFChargedHadrHoP = theTau.hcalEnergyLeadChargedHadrCand() / theTau.leadChargedHadrCand()->p();
    tauVars.leadPFChargedHadrEoP = theTau.ecalEnergyLeadChargedHadrCand() / theTau.leadChargedHadrCand()->p();
  }

  return tauVars;
}

// reco::PFTau
template <class TauType, class ElectronType>
TauVars AntiElectronIDMVA6<TauType, ElectronType>::getTauVarsTypeSpecific(const reco::PFTau& theTau) {  
  TauVars tauVars;
  tauVars.etaAtEcalEntrance = -99.;
  tauVars.leadChargedPFCandEtaAtEcalEntrance = -99.;
  tauVars.leadChargedPFCandPt = -99.;
  float sumEtaTimesEnergy = 0.;
  float sumPhiTimesEnergy = 0.;
  float sumEnergy = 0.;
  tauVars.phi = theTau.phi();
  // Check type of candidates building tau to avoid dynamic casts further 
  bool isFromPFCands = (theTau.leadCand().isNonnull() &&
			dynamic_cast<const reco::PFCandidate*>(theTau.leadCand().get()) != nullptr);
  for (const auto& candidate : theTau.signalCands()) {
    float etaAtECalEntrance = candidate->eta();
    float phiAtECalEntrance = candidate->phi();
    const reco::Track* track = nullptr;
    if (isFromPFCands) {
      const reco::PFCandidate* pfCandidate = static_cast<const reco::PFCandidate*>(candidate.get());
      etaAtECalEntrance = pfCandidate->positionAtECALEntrance().eta();
      if (!usePhiAtEcalEntranceExtrapolation_) {
	phiAtECalEntrance = pfCandidate->positionAtECALEntrance().phi();
      } else {
	bool success = false;
	reco::Candidate::Point posAtECal =
	  positionAtECalEntrance_(candidate.get(), success);
	if (success) {
	  phiAtECalEntrance = posAtECal.phi();
	}
      }
      if (pfCandidate->trackRef().isNonnull())
	track = pfCandidate->trackRef().get();
      else if (pfCandidate->muonRef().isNonnull() &&
	       pfCandidate->muonRef()->innerTrack().isNonnull())
	track = pfCandidate->muonRef()->innerTrack().get();
      else if (pfCandidate->muonRef().isNonnull() &&
	       pfCandidate->muonRef()->globalTrack().isNonnull())
	track = pfCandidate->muonRef()->globalTrack().get();
      else if (pfCandidate->muonRef().isNonnull() &&
	       pfCandidate->muonRef()->outerTrack().isNonnull())
	track = pfCandidate->muonRef()->outerTrack().get();
      else if (pfCandidate->gsfTrackRef().isNonnull())
	track = pfCandidate->gsfTrackRef().get();
    } else {
      bool success = false;
      reco::Candidate::Point posAtECal =
	positionAtECalEntrance_(candidate.get(), success);
      if (success) {
        etaAtECalEntrance = posAtECal.eta();
        phiAtECalEntrance = posAtECal.phi();
      }
      track = candidate->bestTrack();
    }
    if (track != nullptr) {
      if (track->pt() > tauVars.leadChargedPFCandPt) {
        tauVars.leadChargedPFCandEtaAtEcalEntrance = etaAtECalEntrance;
        tauVars.leadChargedPFCandPt = track->pt();
      }
    }
    sumEtaTimesEnergy += etaAtECalEntrance * candidate->energy();
    sumPhiTimesEnergy += phiAtECalEntrance * candidate->energy();
    sumEnergy += candidate->energy();
  }
  if (sumEnergy > 0.) {
    tauVars.etaAtEcalEntrance = sumEtaTimesEnergy / sumEnergy;
    tauVars.phi = sumPhiTimesEnergy / sumEnergy;
  }

  tauVars.emFraction = std::max(theTau.emFraction(), (float)0.);
  tauVars.leadPFChargedHadrHoP = 0.;
  tauVars.leadPFChargedHadrEoP = 0.;
  if (theTau.leadChargedHadrCand()->p() > 0.) {
    if (isFromPFCands) {
      const reco::PFCandidate* pfLeadCandiate =
	static_cast<const reco::PFCandidate*>(theTau.leadChargedHadrCand().get());
      tauVars.leadPFChargedHadrHoP = pfLeadCandiate->hcalEnergy() /
	pfLeadCandiate->p();
      tauVars.leadPFChargedHadrEoP = pfLeadCandiate->ecalEnergy() /
	pfLeadCandiate->p();
    } else {
      const pat::PackedCandidate* patLeadCandiate =
	dynamic_cast<const pat::PackedCandidate*>(theTau.leadChargedHadrCand().get());
      if (patLeadCandiate != nullptr) {
	tauVars.leadPFChargedHadrHoP = patLeadCandiate->caloFraction() *
	  patLeadCandiate->energy() * patLeadCandiate->hcalFraction() /
	  patLeadCandiate->p();
	tauVars.leadPFChargedHadrHoP = patLeadCandiate->caloFraction() *
	  patLeadCandiate->energy() * (1. - patLeadCandiate->hcalFraction()) /
	  patLeadCandiate->p();
      }
    }
  }

  return tauVars;
}

// compile desired types and make available to linker
template class AntiElectronIDMVA6<reco::PFTau, reco::GsfElectron>;
template class AntiElectronIDMVA6<pat::Tau, pat::Electron>;

