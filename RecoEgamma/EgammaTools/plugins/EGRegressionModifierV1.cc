#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "RecoEgamma/EgammaTools/plugins/EGRegressionModifierHelpers.h"
#include "CommonTools/CandAlgos/interface/ModifyObjectValueBase.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoEgamma/EgammaTools/interface/EcalClusterLocal.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include <vdt/vdtMath.h>

class EGRegressionModifierV1 : public ModifyObjectValueBase {
public:
  EGRegressionModifierV1(const edm::ParameterSet& conf, edm::ConsumesCollector& cc);

  void setEvent(const edm::Event&) final;
  void setEventContent(const edm::EventSetup&) final;

  void modifyObject(reco::GsfElectron&) const final;
  void modifyObject(reco::Photon&) const final;

  // just calls reco versions
  void modifyObject(pat::Electron& ele) const final { modifyObject(static_cast<reco::GsfElectron&>(ele)); }
  void modifyObject(pat::Photon& pho) const final { modifyObject(static_cast<reco::Photon&>(pho)); }

private:
  EGRegressionModifierCondTokens eleCond50nsTokens_;
  EGRegressionModifierCondTokens phoCond50nsTokens_;
  EGRegressionModifierCondTokens eleCond25nsTokens_;
  EGRegressionModifierCondTokens phoCond25nsTokens_;

  edm::ESGetToken<GBRForest, GBRWrapperRcd> condNamesWeight50nsToken_;
  edm::ESGetToken<GBRForest, GBRWrapperRcd> condNamesWeight25nsToken_;

  const bool autoDetectBunchSpacing_;
  int bunchspacing_;
  edm::EDGetTokenT<unsigned int> bunchSpacingToken_;
  float rhoValue_;
  edm::EDGetTokenT<double> rhoToken_;
  int nVtx_;
  edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeomToken_;
  CaloGeometry const* caloGeom_;
  const bool applyExtraHighEnergyProtection_;

  std::vector<const GBRForestD*> phoForestsMean_;
  std::vector<const GBRForestD*> phoForestsSigma_;
  std::vector<const GBRForestD*> eleForestsMean_;
  std::vector<const GBRForestD*> eleForestsSigma_;
  const GBRForest* epForest_;
};

DEFINE_EDM_PLUGIN(ModifyObjectValueFactory, EGRegressionModifierV1, "EGRegressionModifierV1");

EGRegressionModifierV1::EGRegressionModifierV1(const edm::ParameterSet& conf, edm::ConsumesCollector& cc)
    : ModifyObjectValueBase(conf),
      eleCond50nsTokens_{conf.getParameterSet("electron_config"), "regressionKey_50ns", "uncertaintyKey_50ns", cc},
      phoCond50nsTokens_{conf.getParameterSet("photon_config"), "regressionKey_50ns", "uncertaintyKey_50ns", cc},
      eleCond25nsTokens_{conf.getParameterSet("electron_config"), "regressionKey_25ns", "uncertaintyKey_25ns", cc},
      phoCond25nsTokens_{conf.getParameterSet("photon_config"), "regressionKey_25ns", "uncertaintyKey_25ns", cc},
      autoDetectBunchSpacing_(conf.getParameter<bool>("autoDetectBunchSpacing")),
      bunchspacing_(autoDetectBunchSpacing_ ? 450 : conf.getParameter<int>("manualBunchSpacing")),
      rhoToken_(cc.consumes(conf.getParameter<edm::InputTag>("rhoCollection"))),
      vtxToken_(cc.consumes(conf.getParameter<edm::InputTag>("vertexCollection"))),
      caloGeomToken_{cc.esConsumes()},
      applyExtraHighEnergyProtection_(conf.getParameter<bool>("applyExtraHighEnergyProtection")) {
  if (autoDetectBunchSpacing_)
    bunchSpacingToken_ = cc.consumes(conf.getParameter<edm::InputTag>("bunchSpacingTag"));

  auto const& electrons = conf.getParameterSet("electron_config");
  condNamesWeight50nsToken_ = cc.esConsumes(electrons.getParameter<edm::ESInputTag>("combinationKey_50ns"));
  condNamesWeight25nsToken_ = cc.esConsumes(electrons.getParameter<edm::ESInputTag>("combinationKey_25ns"));
}

void EGRegressionModifierV1::setEvent(const edm::Event& evt) {
  if (autoDetectBunchSpacing_) {
    bunchspacing_ = evt.get(bunchSpacingToken_);
  }
  rhoValue_ = evt.get(rhoToken_);
  nVtx_ = evt.get(vtxToken_).size();
}

void EGRegressionModifierV1::setEventContent(const edm::EventSetup& evs) {
  caloGeom_ = &evs.getData(caloGeomToken_);

  phoForestsMean_ = retrieveGBRForests(evs, (bunchspacing_ == 25) ? phoCond25nsTokens_.mean : phoCond50nsTokens_.mean);
  phoForestsSigma_ =
      retrieveGBRForests(evs, (bunchspacing_ == 25) ? phoCond25nsTokens_.sigma : phoCond50nsTokens_.sigma);

  eleForestsMean_ = retrieveGBRForests(evs, (bunchspacing_ == 25) ? eleCond25nsTokens_.mean : eleCond50nsTokens_.mean);
  eleForestsSigma_ =
      retrieveGBRForests(evs, (bunchspacing_ == 25) ? eleCond25nsTokens_.sigma : eleCond50nsTokens_.sigma);

  epForest_ = &evs.getData((bunchspacing_ == 25) ? condNamesWeight25nsToken_ : condNamesWeight50nsToken_);
}

void EGRegressionModifierV1::modifyObject(reco::GsfElectron& ele) const {
  // regression calculation needs no additional valuemaps

  const reco::SuperClusterRef& superClus = ele.superCluster();
  const edm::Ptr<reco::CaloCluster>& theseed = superClus->seed();
  const int numberOfClusters = superClus->clusters().size();
  const bool missing_clusters = !superClus->clusters()[numberOfClusters - 1].isAvailable();

  if (missing_clusters)
    return;  // do not apply corrections in case of missing info (slimmed MiniAOD electrons)

  std::array<float, 33> eval;
  const double rawEnergy = superClus->rawEnergy();
  const auto& ess = ele.showerShape();

  // SET INPUTS
  eval[0] = nVtx_;
  eval[1] = rawEnergy;
  eval[2] = superClus->eta();
  eval[3] = superClus->phi();
  eval[4] = superClus->etaWidth();
  eval[5] = superClus->phiWidth();
  eval[6] = ess.r9;
  eval[7] = theseed->energy() / rawEnergy;
  eval[8] = ess.eMax / rawEnergy;
  eval[9] = ess.e2nd / rawEnergy;
  eval[10] = (ess.eLeft + ess.eRight != 0.f ? (ess.eLeft - ess.eRight) / (ess.eLeft + ess.eRight) : 0.f);
  eval[11] = (ess.eTop + ess.eBottom != 0.f ? (ess.eTop - ess.eBottom) / (ess.eTop + ess.eBottom) : 0.f);
  eval[12] = ess.sigmaIetaIeta;
  eval[13] = ess.sigmaIetaIphi;
  eval[14] = ess.sigmaIphiIphi;
  eval[15] = std::max(0, numberOfClusters - 1);

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
  // loop over all clusters that aren't the seed
  for (auto const& pclus : superClus->clusters()) {
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

  eval[16] = clusterMaxDR;
  eval[17] = clusterMaxDRDPhi;
  eval[18] = clusterMaxDRDEta;
  eval[19] = clusterMaxDRRawEnergy / rawEnergy;
  eval[20] = clusterRawEnergy[0] / rawEnergy;
  eval[21] = clusterRawEnergy[1] / rawEnergy;
  eval[22] = clusterRawEnergy[2] / rawEnergy;
  eval[23] = clusterDPhiToSeed[0];
  eval[24] = clusterDPhiToSeed[1];
  eval[25] = clusterDPhiToSeed[2];
  eval[26] = clusterDEtaToSeed[0];
  eval[27] = clusterDEtaToSeed[1];
  eval[28] = clusterDEtaToSeed[2];

  // calculate coordinate variables
  const bool isEB = ele.isEB();
  float dummy;
  int iPhi;
  int iEta;
  float cryPhi;
  float cryEta;
  if (ele.isEB())
    egammaTools::localEcalClusterCoordsEB(*theseed, *caloGeom_, cryEta, cryPhi, iEta, iPhi, dummy, dummy);
  else
    egammaTools::localEcalClusterCoordsEE(*theseed, *caloGeom_, cryEta, cryPhi, iEta, iPhi, dummy, dummy);

  if (isEB) {
    eval[29] = cryEta;
    eval[30] = cryPhi;
    eval[31] = iEta;
    eval[32] = iPhi;
  } else {
    eval[29] = superClus->preshowerEnergy() / superClus->rawEnergy();
  }

  //magic numbers for MINUIT-like transformation of BDT output onto limited range
  //(These should be stored inside the conditions object in the future as well)
  constexpr double meanlimlow = 0.2;
  constexpr double meanlimhigh = 2.0;
  constexpr double meanoffset = meanlimlow + 0.5 * (meanlimhigh - meanlimlow);
  constexpr double meanscale = 0.5 * (meanlimhigh - meanlimlow);

  constexpr double sigmalimlow = 0.0002;
  constexpr double sigmalimhigh = 0.5;
  constexpr double sigmaoffset = sigmalimlow + 0.5 * (sigmalimhigh - sigmalimlow);
  constexpr double sigmascale = 0.5 * (sigmalimhigh - sigmalimlow);

  const int coridx = isEB ? 0 : 1;

  //these are the actual BDT responses
  double rawmean = eleForestsMean_[coridx]->GetResponse(eval.data());
  double rawsigma = eleForestsSigma_[coridx]->GetResponse(eval.data());

  //apply transformation to limited output range (matching the training)
  double mean = meanoffset + meanscale * vdt::fast_sin(rawmean);
  double sigma = sigmaoffset + sigmascale * vdt::fast_sin(rawsigma);

  //regression target is ln(Etrue/Eraw)
  //so corrected energy is ecor=exp(mean)*e, uncertainty is exp(mean)*eraw*sigma=ecor*sigma
  double ecor = mean * (eval[1]);
  if (!isEB)
    ecor = mean * (eval[1] + superClus->preshowerEnergy());
  const double sigmacor = sigma * ecor;

  ele.setCorrectedEcalEnergy(ecor);
  ele.setCorrectedEcalEnergyError(sigmacor);

  // E-p combination
  std::array<float, 11> eval_ep;

  const float ep = ele.trackMomentumAtVtx().R();
  const float tot_energy = superClus->rawEnergy() + superClus->preshowerEnergy();
  const float momentumError = ele.trackMomentumError();
  const float trkMomentumRelError = ele.trackMomentumError() / ep;
  const float eOverP = tot_energy * mean / ep;
  eval_ep[0] = tot_energy * mean;
  eval_ep[1] = sigma / mean;
  eval_ep[2] = ep;
  eval_ep[3] = trkMomentumRelError;
  eval_ep[4] = sigma / mean / trkMomentumRelError;
  eval_ep[5] = tot_energy * mean / ep;
  eval_ep[6] = tot_energy * mean / ep * sqrt(sigma / mean * sigma / mean + trkMomentumRelError * trkMomentumRelError);
  eval_ep[7] = ele.ecalDriven();
  eval_ep[8] = ele.trackerDrivenSeed();
  eval_ep[9] = int(ele.classification());  //eleClass;
  eval_ep[10] = isEB;

  // CODE FOR FUTURE SEMI_PARAMETRIC
  //double rawweight = ep_forestsMean_[coridx]->GetResponse(eval_ep.data());
  ////rawsigma = ep_forestsSigma_[coridx]->GetResponse(eval.data());
  //double weight = meanoffset + meanscale*vdt::fast_sin(rawweight);
  ////sigma = sigmaoffset + sigmascale*vdt::fast_sin(rawsigma);

  // CODE FOR STANDARD BDT
  double weight = 0.;
  if (eOverP > 0.025 && std::abs(ep - ecor) < 15. * std::sqrt(momentumError * momentumError + sigmacor * sigmacor) &&
      (!applyExtraHighEnergyProtection_ || ((momentumError < 10. * ep) || (ecor < 200.)))) {
    // protection against crazy track measurement
    weight = std::clamp(epForest_->GetResponse(eval_ep.data()), 0., 1.);
  }

  double combinedMomentum = weight * ele.trackMomentumAtVtx().R() + (1. - weight) * ecor;
  double combinedMomentumError = sqrt(weight * weight * ele.trackMomentumError() * ele.trackMomentumError() +
                                      (1. - weight) * (1. - weight) * sigmacor * sigmacor);

  math::XYZTLorentzVector oldMomentum = ele.p4();
  math::XYZTLorentzVector newMomentum(oldMomentum.x() * combinedMomentum / oldMomentum.t(),
                                      oldMomentum.y() * combinedMomentum / oldMomentum.t(),
                                      oldMomentum.z() * combinedMomentum / oldMomentum.t(),
                                      combinedMomentum);

  ele.correctMomentum(newMomentum, ele.trackMomentumError(), combinedMomentumError);
}

void EGRegressionModifierV1::modifyObject(reco::Photon& pho) const {
  // regression calculation needs no additional valuemaps

  std::array<float, 35> eval;
  const reco::SuperClusterRef& superClus = pho.superCluster();
  const edm::Ptr<reco::CaloCluster>& theseed = superClus->seed();

  const int numberOfClusters = superClus->clusters().size();
  const bool missing_clusters = !superClus->clusters()[numberOfClusters - 1].isAvailable();

  if (missing_clusters)
    return;  // do not apply corrections in case of missing info (slimmed MiniAOD electrons)

  const double rawEnergy = superClus->rawEnergy();
  const auto& ess = pho.showerShapeVariables();

  // SET INPUTS
  eval[0] = rawEnergy;
  eval[1] = pho.r9();
  eval[2] = superClus->etaWidth();
  eval[3] = superClus->phiWidth();
  eval[4] = std::max(0, numberOfClusters - 1);
  eval[5] = pho.hadronicOverEm();
  eval[6] = rhoValue_;
  eval[7] = nVtx_;
  eval[8] = theseed->eta() - superClus->position().Eta();
  eval[9] = reco::deltaPhi(theseed->phi(), superClus->position().Phi());
  eval[10] = theseed->energy() / rawEnergy;
  eval[11] = ess.e3x3 / ess.e5x5;
  eval[12] = ess.sigmaIetaIeta;
  eval[13] = ess.sigmaIphiIphi;
  eval[14] = ess.sigmaIetaIphi / (ess.sigmaIphiIphi * ess.sigmaIetaIeta);
  eval[15] = ess.maxEnergyXtal / ess.e5x5;
  eval[16] = ess.e2nd / ess.e5x5;
  eval[17] = ess.eTop / ess.e5x5;
  eval[18] = ess.eBottom / ess.e5x5;
  eval[19] = ess.eLeft / ess.e5x5;
  eval[20] = ess.eRight / ess.e5x5;
  eval[21] = ess.e2x5Max / ess.e5x5;
  eval[22] = ess.e2x5Left / ess.e5x5;
  eval[23] = ess.e2x5Right / ess.e5x5;
  eval[24] = ess.e2x5Top / ess.e5x5;
  eval[25] = ess.e2x5Bottom / ess.e5x5;

  const bool isEB = pho.isEB();
  if (isEB) {
    EBDetId ebseedid(theseed->seed());
    eval[26] = pho.e5x5() / theseed->energy();
    int ieta = ebseedid.ieta();
    int iphi = ebseedid.iphi();
    eval[27] = ieta;
    eval[28] = iphi;
    int signieta = ieta > 0 ? +1 : -1;  /// this is 1*abs(ieta)/ieta in original training
    eval[29] = (ieta - signieta) % 5;
    eval[30] = (iphi - 1) % 2;
    eval[31] = (abs(ieta) <= 25) * ((ieta - signieta)) + (abs(ieta) > 25) * ((ieta - 26 * signieta) % 20);
    eval[32] = (iphi - 1) % 20;
    eval[33] = ieta;  /// duplicated variables but this was trained like that
    eval[34] = iphi;  /// duplicated variables but this was trained like that
  } else {
    EEDetId eeseedid(theseed->seed());
    eval[26] = superClus->preshowerEnergy() / rawEnergy;
    eval[27] = superClus->preshowerEnergyPlane1() / rawEnergy;
    eval[28] = superClus->preshowerEnergyPlane2() / rawEnergy;
    eval[29] = eeseedid.ix();
    eval[30] = eeseedid.iy();
  }

  //magic numbers for MINUIT-like transformation of BDT output onto limited range
  //(These should be stored inside the conditions object in the future as well)
  const double meanlimlow = 0.2;
  const double meanlimhigh = 2.0;
  const double meanoffset = meanlimlow + 0.5 * (meanlimhigh - meanlimlow);
  const double meanscale = 0.5 * (meanlimhigh - meanlimlow);

  const double sigmalimlow = 0.0002;
  const double sigmalimhigh = 0.5;
  const double sigmaoffset = sigmalimlow + 0.5 * (sigmalimhigh - sigmalimlow);
  const double sigmascale = 0.5 * (sigmalimhigh - sigmalimlow);

  const int coridx = isEB ? 0 : 1;

  //these are the actual BDT responses
  const double rawmean = phoForestsMean_[coridx]->GetResponse(eval.data());
  const double rawsigma = phoForestsSigma_[coridx]->GetResponse(eval.data());
  //apply transformation to limited output range (matching the training)
  const double mean = meanoffset + meanscale * vdt::fast_sin(rawmean);
  const double sigma = sigmaoffset + sigmascale * vdt::fast_sin(rawsigma);

  //regression target is ln(Etrue/Eraw)
  //so corrected energy is ecor=exp(mean)*e, uncertainty is exp(mean)*eraw*sigma=ecor*sigma
  const double ecor = isEB ? mean * eval[0] : mean * (eval[0] + superClus->preshowerEnergy());

  const double sigmacor = sigma * ecor;
  pho.setCorrectedEnergy(reco::Photon::P4type::regression2, ecor, sigmacor, true);
}
