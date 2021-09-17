#include "RecoEgamma/EgammaElectronAlgos/interface/RegressionHelper.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoEgamma/EgammaTools/interface/EcalRegressionData.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "TVector2.h"
#include "TFile.h"

RegressionHelper::ESGetTokens::ESGetTokens(Configuration const& cfg,
                                           bool useEcalReg,
                                           bool useCombinationReg,
                                           edm::ConsumesCollector& cc)
    : caloTopology{cc.esConsumes()}, caloGeometry{cc.esConsumes()} {
  if (useEcalReg && cfg.ecalWeightsFromDB) {
    ecalRegBarrel = cc.esConsumes<GBRForest, GBRWrapperRcd>(edm::ESInputTag("", cfg.ecalRegressionWeightLabels[0]));
    ecalRegEndcap = cc.esConsumes<GBRForest, GBRWrapperRcd>(edm::ESInputTag("", cfg.ecalRegressionWeightLabels[1]));
    ecalRegErrorBarrel =
        cc.esConsumes<GBRForest, GBRWrapperRcd>(edm::ESInputTag("", cfg.ecalRegressionWeightLabels[2]));
    ecalRegErrorEndcap =
        cc.esConsumes<GBRForest, GBRWrapperRcd>(edm::ESInputTag("", cfg.ecalRegressionWeightLabels[3]));
  }
  if (useCombinationReg && cfg.combinationWeightsFromDB) {
    combinationReg =
        cc.esConsumes<GBRForest, GBRWrapperRcd>(edm::ESInputTag("", cfg.combinationRegressionWeightLabels[0]));
  }
}

RegressionHelper::RegressionHelper(const Configuration& config,
                                   bool useEcalReg,
                                   bool useCombinationReg,
                                   edm::ConsumesCollector& cc)
    : cfg_(config), esGetTokens_{cfg_, useEcalReg, useCombinationReg, cc} {}

void RegressionHelper::applyEcalRegression(reco::GsfElectron& ele,
                                           const reco::VertexCollection& vertices,
                                           const EcalRecHitCollection& rechitsEB,
                                           const EcalRecHitCollection& rechitsEE) const {
  double cor, err;
  getEcalRegression(*ele.superCluster(), vertices, rechitsEB, rechitsEE, cor, err);
  ele.setCorrectedEcalEnergy(cor * ele.superCluster()->correctedEnergy());
  ele.setCorrectedEcalEnergyError(err * ele.superCluster()->correctedEnergy());
}

void RegressionHelper::checkSetup(const edm::EventSetup& es) {
  caloTopology_ = &es.getData(esGetTokens_.caloTopology);
  caloGeometry_ = &es.getData(esGetTokens_.caloGeometry);

  // Ecal regression

  // if at least one of the set of weights come from the DB
  if (cfg_.ecalWeightsFromDB) {
    ecalRegBarrel_ = &es.getData(esGetTokens_.ecalRegBarrel);            // ECAL barrel
    ecalRegEndcap_ = &es.getData(esGetTokens_.ecalRegEndcap);            // ECAL endcaps
    ecalRegErrorBarrel_ = &es.getData(esGetTokens_.ecalRegErrorBarrel);  // ECAL barrel error
    ecalRegErrorEndcap_ = &es.getData(esGetTokens_.ecalRegErrorEndcap);  // ECAL endcap error
    ecalRegressionInitialized_ = true;
  }
  if (cfg_.combinationWeightsFromDB) {
    // Combination
    combinationReg_ = &es.getData(esGetTokens_.combinationReg);
    combinationRegressionInitialized_ = true;
  }

  // read weights from file - for debugging. Even if it is one single files, 4 files should b set in the vector
  if (!cfg_.ecalWeightsFromDB && !ecalRegressionInitialized_ && !cfg_.ecalRegressionWeightFiles.empty()) {
    TFile file0(edm::FileInPath(cfg_.ecalRegressionWeightFiles[0].c_str()).fullPath().c_str());
    ecalRegBarrel_ = (const GBRForest*)file0.Get(cfg_.ecalRegressionWeightLabels[0].c_str());
    file0.Close();
    TFile file1(edm::FileInPath(cfg_.ecalRegressionWeightFiles[1].c_str()).fullPath().c_str());
    ecalRegEndcap_ = (const GBRForest*)file1.Get(cfg_.ecalRegressionWeightLabels[1].c_str());
    file1.Close();
    TFile file2(edm::FileInPath(cfg_.ecalRegressionWeightFiles[2].c_str()).fullPath().c_str());
    ecalRegErrorBarrel_ = (const GBRForest*)file2.Get(cfg_.ecalRegressionWeightLabels[2].c_str());
    file2.Close();
    TFile file3(edm::FileInPath(cfg_.ecalRegressionWeightFiles[3].c_str()).fullPath().c_str());
    ecalRegErrorEndcap_ = (const GBRForest*)file3.Get(cfg_.ecalRegressionWeightLabels[3].c_str());
    ecalRegressionInitialized_ = true;
    file3.Close();
  }

  if (!cfg_.combinationWeightsFromDB && !combinationRegressionInitialized_ &&
      !cfg_.combinationRegressionWeightFiles.empty()) {
    TFile file0(edm::FileInPath(cfg_.combinationRegressionWeightFiles[0].c_str()).fullPath().c_str());
    combinationReg_ = (const GBRForest*)file0.Get(cfg_.combinationRegressionWeightLabels[0].c_str());
    combinationRegressionInitialized_ = true;
    file0.Close();
  }
}

void RegressionHelper::getEcalRegression(const reco::SuperCluster& sc,
                                         const reco::VertexCollection& vertices,
                                         const EcalRecHitCollection& rechitsEB,
                                         const EcalRecHitCollection& rechitsEE,
                                         double& energyFactor,
                                         double& errorFactor) const {
  energyFactor = -999.;
  errorFactor = -999.;

  std::vector<float> rInputs;
  EcalRegressionData regData;
  regData.fill(sc, &rechitsEB, &rechitsEE, caloGeometry_, caloTopology_, &vertices);
  regData.fillVec(rInputs);
  if (sc.seed()->hitsAndFractions()[0].first.subdetId() == EcalBarrel) {
    energyFactor = ecalRegBarrel_->GetResponse(&rInputs[0]);
    errorFactor = ecalRegErrorBarrel_->GetResponse(&rInputs[0]);
  } else if (sc.seed()->hitsAndFractions()[0].first.subdetId() == EcalEndcap) {
    energyFactor = ecalRegEndcap_->GetResponse(&rInputs[0]);
    errorFactor = ecalRegErrorEndcap_->GetResponse(&rInputs[0]);
  } else {
    throw cms::Exception("RegressionHelper::calculateRegressedEnergy")
        << "Supercluster seed is either EB nor EE!" << std::endl;
  }
}

void RegressionHelper::applyCombinationRegression(reco::GsfElectron& ele) const {
  float energy = ele.correctedEcalEnergy();
  float energyError = ele.correctedEcalEnergyError();
  float momentum = ele.trackMomentumAtVtx().R();
  float momentumError = ele.trackMomentumError();
  int elClass = -1;

  switch (ele.classification()) {
    case reco::GsfElectron::GOLDEN:
      elClass = 0;
      break;
    case reco::GsfElectron::BIGBREM:
      elClass = 1;
      break;
    case reco::GsfElectron::BADTRACK:
      elClass = 2;
      break;
    case reco::GsfElectron::SHOWERING:
      elClass = 3;
      break;
    case reco::GsfElectron::GAP:
      elClass = 4;
      break;
    default:
      elClass = -1;
  }

  bool isEcalDriven = ele.ecalDriven();
  bool isTrackerDriven = ele.trackerDrivenSeed();
  bool isEB = ele.isEB();

  // compute relative errors and ratio of errors
  float energyRelError = energyError / energy;
  float momentumRelError = momentumError / momentum;
  float errorRatio = energyRelError / momentumRelError;

  // calculate E/p and corresponding error
  float eOverP = energy / momentum;
  float eOverPerror = eOverP * std::hypot(energyRelError, momentumRelError);

  // fill input variables
  std::vector<float> regressionInputs;
  regressionInputs.resize(11, 0.);

  regressionInputs[0] = energy;
  regressionInputs[1] = energyRelError;
  regressionInputs[2] = momentum;
  regressionInputs[3] = momentumRelError;
  regressionInputs[4] = errorRatio;
  regressionInputs[5] = eOverP;
  regressionInputs[6] = eOverPerror;
  regressionInputs[7] = static_cast<float>(isEcalDriven);
  regressionInputs[8] = static_cast<float>(isTrackerDriven);
  regressionInputs[9] = static_cast<float>(elClass);
  regressionInputs[10] = static_cast<float>(isEB);

  // retrieve combination weight
  float weight = 0.;
  if (eOverP > 0.025 &&
      fabs(momentum - energy) < 15. * sqrt(momentumError * momentumError +
                                           energyError * energyError))  // protection against crazy track measurement
  {
    weight = combinationReg_->GetResponse(regressionInputs.data());
    if (weight > 1.)
      weight = 1.;
    else if (weight < 0.)
      weight = 0.;
  }

  float combinedMomentum = weight * momentum + (1. - weight) * energy;
  float combinedMomentumError =
      sqrt(weight * weight * momentumError * momentumError + (1. - weight) * (1. - weight) * energyError * energyError);

  // FIXME : pure tracker electrons have track momentum error of 999.
  // If the combination try to combine such electrons then the original combined momentum is kept
  if (momentumError != 999. || weight == 0.) {
    math::XYZTLorentzVector oldMomentum = ele.p4();
    math::XYZTLorentzVector newMomentum(oldMomentum.x() * combinedMomentum / oldMomentum.t(),
                                        oldMomentum.y() * combinedMomentum / oldMomentum.t(),
                                        oldMomentum.z() * combinedMomentum / oldMomentum.t(),
                                        combinedMomentum);

    ele.setP4(reco::GsfElectron::P4_COMBINATION, newMomentum, combinedMomentumError, true);
  }
}
