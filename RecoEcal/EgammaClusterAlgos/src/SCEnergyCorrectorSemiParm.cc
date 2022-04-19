#include "RecoEcal/EgammaClusterAlgos/interface/SCEnergyCorrectorSemiParm.h"

#include "FWCore/Utilities/interface/isFinite.h"
#include "FWCore/Utilities/interface/Transition.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalTools.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoEgamma/EgammaTools/interface/EgammaHGCALIDParamDefaults.h"

#include <vdt/vdtMath.h>

using namespace reco;

namespace {
  //bool is if a valid dr was found, float is the dr
  std::pair<bool, float> getMaxDRNonSeedCluster(const reco::SuperCluster& sc) {
    float maxDR2 = 0.;
    const edm::Ptr<reco::CaloCluster>& seedClus = sc.seed();

    for (const auto& clus : sc.clusters()) {
      if (clus == seedClus) {
        continue;
      }

      // find cluster with max dR
      const double dr2 = reco::deltaR2(*clus, *seedClus);
      if (dr2 > maxDR2) {
        maxDR2 = dr2;
      }
    }
    return {sc.clustersSize() != 1, sc.clustersSize() != 1 ? std::sqrt(maxDR2) : 999.};
  }
  template <typename T>
  int countRecHits(const T& recHitHandle, float threshold) {
    int count = 0;
    if (recHitHandle.isValid()) {
      for (const auto& recHit : *recHitHandle) {
        if (recHit.energy() > threshold) {
          count++;
        }
      }
    }
    return count;
  }
}  // namespace

SCEnergyCorrectorSemiParm::SCEnergyCorrectorSemiParm()
    : caloTopo_(nullptr),
      caloGeom_(nullptr),
      isHLT_(false),
      isPhaseII_(false),
      regTrainedWithPS_(true),
      applySigmaIetaIphiBug_(false),
      nHitsAboveThresholdEB_(0),
      nHitsAboveThresholdEE_(0),
      nHitsAboveThresholdHG_(0),
      hitsEnergyThreshold_(-1.),
      hgcalCylinderR_(0.) {}

SCEnergyCorrectorSemiParm::SCEnergyCorrectorSemiParm(const edm::ParameterSet& iConfig, edm::ConsumesCollector cc)
    : SCEnergyCorrectorSemiParm() {
  setTokens(iConfig, cc);
}

void SCEnergyCorrectorSemiParm::fillPSetDescription(edm::ParameterSetDescription& desc) {
  desc.add<bool>("isHLT", false);
  desc.add<bool>("isPhaseII", false);
  desc.add<bool>("regTrainedWithPS", true);
  desc.add<bool>("applySigmaIetaIphiBug", false);
  desc.add<edm::InputTag>("ecalRecHitsEE", edm::InputTag("ecalRecHit", "EcalRecHitsEE"));
  desc.add<edm::InputTag>("ecalRecHitsEB", edm::InputTag("ecalRecHit", "EcalRecHitsEB"));
  desc.add<std::string>("regressionKeyEB", "pfscecal_EBCorrection_offline_v2");
  desc.add<std::string>("regressionKeyEE", "pfscecal_EECorrection_offline_v2");
  desc.add<std::string>("uncertaintyKeyEB", "pfscecal_EBUncertainty_offline_v2");
  desc.add<std::string>("uncertaintyKeyEE", "pfscecal_EEUncertainty_offline_v2");
  desc.add<double>("regressionMinEB", 0.2);
  desc.add<double>("regressionMaxEB", 2.0);
  desc.add<double>("regressionMinEE", 0.2);
  desc.add<double>("regressionMaxEE", 2.0);
  desc.add<double>("uncertaintyMinEB", 0.0002);
  desc.add<double>("uncertaintyMaxEB", 0.5);
  desc.add<double>("uncertaintyMinEE", 0.0002);
  desc.add<double>("uncertaintyMaxEE", 0.5);
  desc.add<edm::InputTag>("vertexCollection", edm::InputTag("offlinePrimaryVertices"));
  desc.add<double>("eRecHitThreshold", 1.);
  desc.add<edm::InputTag>("hgcalRecHits", edm::InputTag());
  desc.add<double>("hgcalCylinderR", EgammaHGCALIDParamDefaults::kRCylinder);
}

edm::ParameterSetDescription SCEnergyCorrectorSemiParm::makePSetDescription() {
  edm::ParameterSetDescription desc;
  fillPSetDescription(desc);
  return desc;
}

void SCEnergyCorrectorSemiParm::setEventSetup(const edm::EventSetup& es) {
  caloTopo_ = &es.getData(caloTopoToken_);
  caloGeom_ = &es.getData(caloGeomToken_);

  regParamBarrel_.setForests(es);
  regParamEndcap_.setForests(es);

  if (isPhaseII_) {
    hgcalShowerShapes_.initPerSetup(es);
  }
}

void SCEnergyCorrectorSemiParm::setEvent(const edm::Event& event) {
  event.getByToken(tokenEBRecHits_, recHitsEB_);
  if (!isPhaseII_) {
    event.getByToken(tokenEERecHits_, recHitsEE_);
  } else {
    event.getByToken(tokenHgcalRecHits_, recHitsHgcal_);
    hgcalShowerShapes_.initPerEvent(*recHitsHgcal_);
  }
  if (isHLT_ || isPhaseII_) {
    //note countRecHits checks the validity of the handle and returns 0
    //if invalid so its okay to call on all rec-hit collections here
    nHitsAboveThresholdEB_ = countRecHits(recHitsEB_, hitsEnergyThreshold_);
    nHitsAboveThresholdEE_ = countRecHits(recHitsEE_, hitsEnergyThreshold_);
    nHitsAboveThresholdHG_ = countRecHits(recHitsHgcal_, hitsEnergyThreshold_);
  }
  if (!isHLT_) {
    event.getByToken(tokenVertices_, vertices_);
  }
}

std::pair<double, double> SCEnergyCorrectorSemiParm::getCorrections(const reco::SuperCluster& sc) const {
  std::pair<double, double> corrEnergyAndRes = {-1, -1};

  const auto regData = getRegData(sc);
  if (regData.empty()) {
    //supercluster has no valid regression, return default values
    return corrEnergyAndRes;
  }
  DetId seedId = sc.seed()->seed();
  const auto& regParam = getRegParam(seedId);

  double mean = regParam.mean(regData);
  double sigma = regParam.sigma(regData);

  double energyCorr = mean * sc.rawEnergy();
  if (isHLT_ && sc.seed()->seed().det() == DetId::Ecal && seedId.subdetId() == EcalEndcap && !regTrainedWithPS_) {
    energyCorr += sc.preshowerEnergy();
  }
  double resolutionEst = sigma * energyCorr;

  corrEnergyAndRes.first = energyCorr;
  corrEnergyAndRes.second = resolutionEst;

  return corrEnergyAndRes;
}

void SCEnergyCorrectorSemiParm::modifyObject(reco::SuperCluster& sc) const {
  std::pair<double, double> cor = getCorrections(sc);
  if (cor.first < 0)
    return;
  sc.setEnergy(cor.first);
  sc.setCorrectedEnergy(cor.first);
  if (cor.second >= 0) {
    sc.setCorrectedEnergyUncertainty(cor.second);
  }
}

std::vector<float> SCEnergyCorrectorSemiParm::getRegData(const reco::SuperCluster& sc) const {
  switch (sc.seed()->seed().det()) {
    case DetId::Ecal:
      if (isPhaseII_ && sc.seed()->seed().subdetId() == EcalEndcap) {
        throw cms::Exception("ConfigError") << " Error in SCEnergyCorrectorSemiParm: "
                                            << " running over events with EcalEndcap clusters while enabling "
                                               "isPhaseII, please set isPhaseII = False in regression config";
      }
      return isHLT_ ? getRegDataECALHLTV1(sc) : getRegDataECALV1(sc);
    case DetId::HGCalEE:
      if (!isPhaseII_) {
        throw cms::Exception("ConfigError") << " Error in SCEnergyCorrectorSemiParm: "
                                            << " running over PhaseII events without enabling isPhaseII, please set "
                                               "isPhaseII = True in regression config";
      }
      return isHLT_ ? getRegDataHGCALHLTV1(sc) : getRegDataHGCALV1(sc);
    default:
      return std::vector<float>();
  }
}

void SCEnergyCorrectorSemiParm::RegParam::setForests(const edm::EventSetup& setup) {
  meanForest_ = &setup.getData(meanForestToken_);
  sigmaForest_ = &setup.getData(sigmaForestToken_);
}

double SCEnergyCorrectorSemiParm::RegParam::mean(const std::vector<float>& data) const {
  return meanForest_ ? meanOutTrans_(meanForest_->GetResponse(data.data())) : -1;
}

double SCEnergyCorrectorSemiParm::RegParam::sigma(const std::vector<float>& data) const {
  return sigmaForest_ ? sigmaOutTrans_(sigmaForest_->GetResponse(data.data())) : -1;
}

std::vector<float> SCEnergyCorrectorSemiParm::getRegDataECALV1(const reco::SuperCluster& sc) const {
  std::vector<float> eval(30, 0.);

  const reco::CaloCluster& seedCluster = *(sc.seed());
  const bool iseb = seedCluster.hitsAndFractions()[0].first.subdetId() == EcalBarrel;
  const EcalRecHitCollection* recHits = iseb ? recHitsEB_.product() : recHitsEE_.product();

  const double raw_energy = sc.rawEnergy();
  const int numberOfClusters = sc.clusters().size();

  const auto& localCovariances = EcalClusterTools::localCovariances(seedCluster, recHits, caloTopo_);

  const float eLeft = EcalClusterTools::eLeft(seedCluster, recHits, caloTopo_);
  const float eRight = EcalClusterTools::eRight(seedCluster, recHits, caloTopo_);
  const float eTop = EcalClusterTools::eTop(seedCluster, recHits, caloTopo_);
  const float eBottom = EcalClusterTools::eBottom(seedCluster, recHits, caloTopo_);

  float sigmaIetaIeta = sqrt(localCovariances[0]);
  float sigmaIetaIphi = std::numeric_limits<float>::max();
  float sigmaIphiIphi = std::numeric_limits<float>::max();

  if (!edm::isNotFinite(localCovariances[2]))
    sigmaIphiIphi = sqrt(localCovariances[2]);

  // extra shower shapes
  const float see_by_spp = sigmaIetaIeta * (applySigmaIetaIphiBug_ ? std::numeric_limits<float>::max() : sigmaIphiIphi);
  if (see_by_spp > 0) {
    sigmaIetaIphi = localCovariances[1] / see_by_spp;
  } else if (localCovariances[1] > 0) {
    sigmaIetaIphi = 1.f;
  } else {
    sigmaIetaIphi = -1.f;
  }

  // calculate sub-cluster variables
  std::vector<float> clusterRawEnergy;
  clusterRawEnergy.resize(std::max(3, numberOfClusters), 0);
  std::vector<float> clusterDEtaToSeed;
  clusterDEtaToSeed.resize(std::max(3, numberOfClusters), 0);
  std::vector<float> clusterDPhiToSeed;
  clusterDPhiToSeed.resize(std::max(3, numberOfClusters), 0);
  float clusterMaxDR = 999.;
  float clusterMaxDRDPhi = 999.;
  float clusterMaxDRDEta = 999.;
  float clusterMaxDRRawEnergy = 0.;

  size_t iclus = 0;
  float maxDR = 0;
  edm::Ptr<reco::CaloCluster> pclus;
  const edm::Ptr<reco::CaloCluster>& theseed = sc.seed();
  // loop over all clusters that aren't the seed
  auto clusend = sc.clustersEnd();
  for (auto clus = sc.clustersBegin(); clus != clusend; ++clus) {
    pclus = *clus;

    if (theseed == pclus)
      continue;
    clusterRawEnergy[iclus] = pclus->energy();
    clusterDPhiToSeed[iclus] = reco::deltaPhi(pclus->phi(), theseed->phi());
    clusterDEtaToSeed[iclus] = pclus->eta() - theseed->eta();

    // find cluster with max dR
    const auto the_dr = reco::deltaR(*pclus, *theseed);
    if (the_dr > maxDR) {
      maxDR = the_dr;
      clusterMaxDR = maxDR;
      clusterMaxDRDPhi = clusterDPhiToSeed[iclus];
      clusterMaxDRDEta = clusterDEtaToSeed[iclus];
      clusterMaxDRRawEnergy = clusterRawEnergy[iclus];
    }
    ++iclus;
  }

  eval[0] = vertices_->size();
  eval[1] = raw_energy;
  eval[2] = sc.etaWidth();
  eval[3] = sc.phiWidth();
  eval[4] = EcalClusterTools::e3x3(seedCluster, recHits, caloTopo_) / raw_energy;
  eval[5] = seedCluster.energy() / raw_energy;
  eval[6] = EcalClusterTools::eMax(seedCluster, recHits) / raw_energy;
  eval[7] = EcalClusterTools::e2nd(seedCluster, recHits) / raw_energy;
  eval[8] = (eLeft + eRight != 0.f ? (eLeft - eRight) / (eLeft + eRight) : 0.f);
  eval[9] = (eTop + eBottom != 0.f ? (eTop - eBottom) / (eTop + eBottom) : 0.f);
  eval[10] = sigmaIetaIeta;
  eval[11] = sigmaIetaIphi;
  eval[12] = sigmaIphiIphi;
  eval[13] = std::max(0, numberOfClusters - 1);
  eval[14] = clusterMaxDR;
  eval[15] = clusterMaxDRDPhi;
  eval[16] = clusterMaxDRDEta;
  eval[17] = clusterMaxDRRawEnergy / raw_energy;
  eval[18] = clusterRawEnergy[0] / raw_energy;
  eval[19] = clusterRawEnergy[1] / raw_energy;
  eval[20] = clusterRawEnergy[2] / raw_energy;
  eval[21] = clusterDPhiToSeed[0];
  eval[22] = clusterDPhiToSeed[1];
  eval[23] = clusterDPhiToSeed[2];
  eval[24] = clusterDEtaToSeed[0];
  eval[25] = clusterDEtaToSeed[1];
  eval[26] = clusterDEtaToSeed[2];
  if (iseb) {
    EBDetId ebseedid(seedCluster.seed());
    eval[27] = ebseedid.ieta();
    eval[28] = ebseedid.iphi();
  } else {
    EEDetId eeseedid(seedCluster.seed());
    eval[27] = eeseedid.ix();
    eval[28] = eeseedid.iy();
    //seed cluster eta is only needed for the 106X Ultra Legacy regressions
    //and was not used in the 74X regression however as its just an extra varaible
    //at the end, its harmless to add for the 74X regression
    eval[29] = seedCluster.eta();
  }
  return eval;
}

std::vector<float> SCEnergyCorrectorSemiParm::getRegDataECALHLTV1(const reco::SuperCluster& sc) const {
  std::vector<float> eval(7, 0.);
  auto maxDRNonSeedClus = getMaxDRNonSeedCluster(sc);
  const float clusterMaxDR = maxDRNonSeedClus.first ? maxDRNonSeedClus.second : 999.;

  const reco::CaloCluster& seedCluster = *(sc.seed());
  const bool iseb = seedCluster.hitsAndFractions()[0].first.subdetId() == EcalBarrel;
  const EcalRecHitCollection* recHits = iseb ? recHitsEB_.product() : recHitsEE_.product();

  eval[0] = nHitsAboveThresholdEB_ + nHitsAboveThresholdEE_;
  eval[1] = sc.eta();
  eval[2] = sc.phiWidth();
  eval[3] = EcalClusterTools::e3x3(seedCluster, recHits, caloTopo_) / sc.rawEnergy();
  eval[4] = std::max(0, static_cast<int>(sc.clusters().size()) - 1);
  eval[5] = clusterMaxDR;
  eval[6] = sc.rawEnergy();

  return eval;
}

std::vector<float> SCEnergyCorrectorSemiParm::getRegDataHGCALV1(const reco::SuperCluster& sc) const {
  std::vector<float> eval(17, 0.);

  auto ssCalc = hgcalShowerShapes_.createCalc(sc);
  auto pcaWidths = ssCalc.getPCAWidths(hgcalCylinderR_);
  auto energyHighestHits = ssCalc.getEnergyHighestHits(2);

  auto maxDRNonSeedClus = getMaxDRNonSeedCluster(sc);
  const float clusterMaxDR = maxDRNonSeedClus.first ? maxDRNonSeedClus.second : 999.;

  eval[0] = sc.rawEnergy();
  eval[1] = sc.eta();
  eval[2] = sc.etaWidth();
  eval[3] = sc.phiWidth();
  eval[4] = sc.clusters().size();
  eval[5] = sc.hitsAndFractions().size();
  eval[6] = clusterMaxDR;
  eval[7] = sc.eta() - sc.seed()->eta();
  eval[8] = reco::deltaPhi(sc.phi(), sc.seed()->phi());
  eval[9] = energyHighestHits[0] / sc.rawEnergy();
  eval[10] = energyHighestHits[1] / sc.rawEnergy();
  eval[11] = std::sqrt(pcaWidths.sigma2uu);
  eval[12] = std::sqrt(pcaWidths.sigma2vv);
  eval[13] = std::sqrt(pcaWidths.sigma2ww);
  eval[14] = ssCalc.getRvar(hgcalCylinderR_, sc.rawEnergy());
  eval[15] = sc.seed()->energy() / sc.rawEnergy();
  eval[16] = nHitsAboveThresholdEB_ + nHitsAboveThresholdHG_;

  return eval;
}

std::vector<float> SCEnergyCorrectorSemiParm::getRegDataHGCALHLTV1(const reco::SuperCluster& sc) const {
  std::vector<float> eval(7, 0.);
  const float clusterMaxDR = getMaxDRNonSeedCluster(sc).second;

  auto ssCalc = hgcalShowerShapes_.createCalc(sc);

  eval[0] = sc.rawEnergy();
  eval[1] = sc.eta();
  eval[2] = sc.phiWidth();
  eval[3] = std::max(0, static_cast<int>(sc.clusters().size()) - 1);
  eval[4] = ssCalc.getRvar(hgcalCylinderR_);
  eval[5] = clusterMaxDR;
  eval[6] = nHitsAboveThresholdEB_ + nHitsAboveThresholdHG_;

  return eval;
}
