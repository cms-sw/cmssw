#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalTools.h"
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

class EGRegressionModifierV2 : public ModifyObjectValueBase {
public:
  EGRegressionModifierV2(const edm::ParameterSet& conf, edm::ConsumesCollector& cc);

  void setEvent(const edm::Event&) final;
  void setEventContent(const edm::EventSetup&) final;

  void modifyObject(reco::GsfElectron&) const final;
  void modifyObject(reco::Photon&) const final;

  // just calls reco versions
  void modifyObject(pat::Electron& ele) const final { modifyObject(static_cast<reco::GsfElectron&>(ele)); }
  void modifyObject(pat::Photon& pho) const final { modifyObject(static_cast<reco::Photon&>(pho)); }

private:
  EGRegressionModifierCondTokens eleCondTokens_;
  EGRegressionModifierCondTokens phoCondTokens_;

  float rhoValue_;
  const edm::EDGetTokenT<double> rhoToken_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryToken_;
  CaloGeometry const* caloGeometry_ = nullptr;

  std::vector<const GBRForestD*> phoForestsMean_;
  std::vector<const GBRForestD*> phoForestsSigma_;
  std::vector<const GBRForestD*> eleForestsMean_;
  std::vector<const GBRForestD*> eleForestsSigma_;

  const double lowEnergyEcalOnlyThr_;    // 300
  const double lowEnergyEcalTrackThr_;   // 50
  const double highEnergyEcalTrackThr_;  // 200
  const double eOverPEcalTrkThr_;        // 0.025
  const double epDiffSigEcalTrackThr_;   // 15
  const double epSigEcalTrackThr_;       // 10
  const bool forceHighEnergyEcalTrainingIfSaturated_;
};

DEFINE_EDM_PLUGIN(ModifyObjectValueFactory, EGRegressionModifierV2, "EGRegressionModifierV2");

EGRegressionModifierV2::EGRegressionModifierV2(const edm::ParameterSet& conf, edm::ConsumesCollector& cc)
    : ModifyObjectValueBase(conf),
      eleCondTokens_{conf.getParameterSet("electron_config"), "regressionKey", "uncertaintyKey", cc},
      phoCondTokens_{conf.getParameterSet("photon_config"), "regressionKey", "uncertaintyKey", cc},

      rhoToken_(cc.consumes(conf.getParameter<edm::InputTag>("rhoCollection"))),
      caloGeometryToken_(cc.esConsumes()),
      lowEnergyEcalOnlyThr_(conf.getParameter<double>("lowEnergy_ECALonlyThr")),
      lowEnergyEcalTrackThr_(conf.getParameter<double>("lowEnergy_ECALTRKThr")),
      highEnergyEcalTrackThr_(conf.getParameter<double>("highEnergy_ECALTRKThr")),
      eOverPEcalTrkThr_(conf.getParameter<double>("eOverP_ECALTRKThr")),
      epDiffSigEcalTrackThr_(conf.getParameter<double>("epDiffSig_ECALTRKThr")),
      epSigEcalTrackThr_(conf.getParameter<double>("epSig_ECALTRKThr")),
      forceHighEnergyEcalTrainingIfSaturated_(conf.getParameter<bool>("forceHighEnergyEcalTrainingIfSaturated")) {
  unsigned int encor = eleCondTokens_.mean.size();
  eleForestsMean_.reserve(2 * encor);
  eleForestsSigma_.reserve(2 * encor);

  unsigned int ncor = phoCondTokens_.mean.size();
  phoForestsMean_.reserve(ncor);
  phoForestsSigma_.reserve(ncor);
}

void EGRegressionModifierV2::setEvent(const edm::Event& evt) { rhoValue_ = evt.get(rhoToken_); }

void EGRegressionModifierV2::setEventContent(const edm::EventSetup& evs) {
  phoForestsMean_ = retrieveGBRForests(evs, phoCondTokens_.mean);
  phoForestsSigma_ = retrieveGBRForests(evs, phoCondTokens_.sigma);

  eleForestsMean_ = retrieveGBRForests(evs, eleCondTokens_.mean);
  eleForestsSigma_ = retrieveGBRForests(evs, eleCondTokens_.sigma);

  caloGeometry_ = &evs.getData(caloGeometryToken_);
}

void EGRegressionModifierV2::modifyObject(reco::GsfElectron& ele) const {
  // regression calculation needs no additional valuemaps

  const reco::SuperClusterRef& superClus = ele.superCluster();
  const edm::Ptr<reco::CaloCluster>& seed = superClus->seed();

  // skip HGCAL for now
  if (EcalTools::isHGCalDet(seed->seed().det()))
    return;

  const int numberOfClusters = superClus->clusters().size();
  const bool missing_clusters = !superClus->clusters()[numberOfClusters - 1].isAvailable();
  if (missing_clusters)
    return;  // do not apply corrections in case of missing info (slimmed MiniAOD electrons)

  //check if fbrem is filled as its needed for E/p combination so abort if its set to the default value
  //this will be the case for <5 (or current cuts) for miniAOD electrons
  if (ele.fbrem() == reco::GsfElectron::ClassificationVariables().trackFbrem)
    return;

  const bool isEB = ele.isEB();

  std::array<float, 32> eval;
  const double rawEnergy = superClus->rawEnergy();
  const double raw_es_energy = superClus->preshowerEnergy();
  const auto& full5x5_ess = ele.full5x5_showerShape();

  float e5x5Inverse = full5x5_ess.e5x5 != 0. ? vdt::fast_inv(full5x5_ess.e5x5) : 0.;

  eval[0] = rawEnergy;
  eval[1] = superClus->etaWidth();
  eval[2] = superClus->phiWidth();
  eval[3] = superClus->seed()->energy() / rawEnergy;
  eval[4] = full5x5_ess.e5x5 / rawEnergy;
  eval[5] = ele.hcalOverEcalBc();
  eval[6] = rhoValue_;
  eval[7] = seed->eta() - superClus->position().Eta();
  eval[8] = reco::deltaPhi(seed->phi(), superClus->position().Phi());
  eval[9] = full5x5_ess.r9;
  eval[10] = full5x5_ess.sigmaIetaIeta;
  eval[11] = full5x5_ess.sigmaIetaIphi;
  eval[12] = full5x5_ess.sigmaIphiIphi;
  eval[13] = full5x5_ess.eMax * e5x5Inverse;
  eval[14] = full5x5_ess.e2nd * e5x5Inverse;
  eval[15] = full5x5_ess.eTop * e5x5Inverse;
  eval[16] = full5x5_ess.eBottom * e5x5Inverse;
  eval[17] = full5x5_ess.eLeft * e5x5Inverse;
  eval[18] = full5x5_ess.eRight * e5x5Inverse;
  eval[19] = full5x5_ess.e2x5Max * e5x5Inverse;
  eval[20] = full5x5_ess.e2x5Left * e5x5Inverse;
  eval[21] = full5x5_ess.e2x5Right * e5x5Inverse;
  eval[22] = full5x5_ess.e2x5Top * e5x5Inverse;
  eval[23] = full5x5_ess.e2x5Bottom * e5x5Inverse;
  eval[24] = ele.nSaturatedXtals();
  eval[25] = std::max(0, numberOfClusters);

  // calculate coordinate variables
  if (isEB) {
    float dummy;
    int ieta;
    int iphi;
    egammaTools::localEcalClusterCoordsEB(*seed, *caloGeometry_, dummy, dummy, ieta, iphi, dummy, dummy);
    eval[26] = ieta;
    eval[27] = iphi;
    int signieta = ieta > 0 ? +1 : -1;
    eval[28] = (ieta - signieta) % 5;
    eval[29] = (iphi - 1) % 2;
    eval[30] = (abs(ieta) <= 25) * ((ieta - signieta)) + (abs(ieta) > 25) * ((ieta - 26 * signieta) % 20);
    eval[31] = (iphi - 1) % 20;

  } else {
    float dummy;
    int ix;
    int iy;
    egammaTools::localEcalClusterCoordsEE(*seed, *caloGeometry_, dummy, dummy, ix, iy, dummy, dummy);
    eval[26] = ix;
    eval[27] = iy;
    eval[28] = raw_es_energy / rawEnergy;
  }

  //magic numbers for MINUIT-like transformation of BDT output onto limited range
  //(These should be stored inside the conditions object in the future as well)
  constexpr double meanlimlow = -1.0;
  constexpr double meanlimhigh = 3.0;
  constexpr double meanoffset = meanlimlow + 0.5 * (meanlimhigh - meanlimlow);
  constexpr double meanscale = 0.5 * (meanlimhigh - meanlimlow);

  constexpr double sigmalimlow = 0.0002;
  constexpr double sigmalimhigh = 0.5;
  constexpr double sigmaoffset = sigmalimlow + 0.5 * (sigmalimhigh - sigmalimlow);
  constexpr double sigmascale = 0.5 * (sigmalimhigh - sigmalimlow);

  size_t coridx = 0;
  float rawPt = rawEnergy * superClus->position().rho() / superClus->position().r();
  bool isSaturated = ele.nSaturatedXtals() != 0;

  if (rawPt >= lowEnergyEcalOnlyThr_ || (isSaturated && forceHighEnergyEcalTrainingIfSaturated_)) {
    if (isEB)
      coridx = 1;
    else
      coridx = 3;
  } else {
    if (isEB)
      coridx = 0;
    else
      coridx = 2;
  }

  //these are the actual BDT responses
  double rawmean = eleForestsMean_[coridx]->GetResponse(eval.data());
  double rawsigma = eleForestsSigma_[coridx]->GetResponse(eval.data());

  //apply transformation to limited output range (matching the training)
  double mean = meanoffset + meanscale * vdt::fast_sin(rawmean);
  double sigma = sigmaoffset + sigmascale * vdt::fast_sin(rawsigma);

  // Correct the energy. A negative energy means that the correction went
  // outside the boundaries of the training. In this case uses raw.
  // The resolution estimation, on the other hand should be ok.
  if (mean < 0.)
    mean = 1.0;

  const double ecor = mean * (rawEnergy + raw_es_energy);
  const double sigmacor = sigma * ecor;

  ele.setCorrectedEcalEnergy(ecor);
  ele.setCorrectedEcalEnergyError(sigmacor);

  double combinedEnergy = ecor;
  double combinedEnergyError = sigmacor;

  auto el_track = ele.gsfTrack();
  const float trkMomentum = el_track->pMode();
  const float trkEta = el_track->etaMode();
  const float trkPhi = el_track->phiMode();
  const float trkMomentumError = std::abs(el_track->qoverpModeError()) * trkMomentum * trkMomentum;

  const float eOverP = (rawEnergy + raw_es_energy) * mean / trkMomentum;
  const float fbrem = ele.fbrem();

  // E-p combination
  if (ecor < highEnergyEcalTrackThr_ && eOverP > eOverPEcalTrkThr_ &&
      std::abs(ecor - trkMomentum) <
          epDiffSigEcalTrackThr_ * std::sqrt(trkMomentumError * trkMomentumError + sigmacor * sigmacor) &&
      trkMomentumError < epSigEcalTrackThr_ * trkMomentum) {
    rawPt = ecor / cosh(trkEta);
    if (isEB && rawPt < lowEnergyEcalTrackThr_)
      coridx = 4;
    else if (isEB && rawPt >= lowEnergyEcalTrackThr_)
      coridx = 5;
    else if (!isEB && rawPt < lowEnergyEcalTrackThr_)
      coridx = 6;
    else if (!isEB && rawPt >= lowEnergyEcalTrackThr_)
      coridx = 7;

    eval[0] = ecor;
    eval[1] = sigma / mean;
    eval[2] = trkMomentumError / trkMomentum;
    eval[3] = eOverP;
    eval[4] = ele.ecalDrivenSeed();
    eval[5] = full5x5_ess.r9;
    eval[6] = fbrem;
    eval[7] = trkEta;
    eval[8] = trkPhi;

    float ecalEnergyVar = (rawEnergy + raw_es_energy) * sigma;
    float rawcombNormalization = (trkMomentumError * trkMomentumError + ecalEnergyVar * ecalEnergyVar);
    float rawcomb = (ecor * trkMomentumError * trkMomentumError + trkMomentum * ecalEnergyVar * ecalEnergyVar) /
                    rawcombNormalization;

    //these are the actual BDT responses
    double rawmean_trk = eleForestsMean_[coridx]->GetResponse(eval.data());
    double rawsigma_trk = eleForestsSigma_[coridx]->GetResponse(eval.data());

    //apply transformation to limited output range (matching the training)
    double mean_trk = meanoffset + meanscale * vdt::fast_sin(rawmean_trk);
    double sigma_trk = sigmaoffset + sigmascale * vdt::fast_sin(rawsigma_trk);

    // Final correction
    // A negative energy means that the correction went
    // outside the boundaries of the training. In this case uses raw.
    // The resolution estimation, on the other hand should be ok.
    if (mean_trk < 0.)
      mean_trk = 1.0;

    combinedEnergy = mean_trk * rawcomb;
    combinedEnergyError = sigma_trk * rawcomb;
  }

  math::XYZTLorentzVector oldFourMomentum = ele.p4();
  math::XYZTLorentzVector newFourMomentum(oldFourMomentum.x() * combinedEnergy / oldFourMomentum.t(),
                                          oldFourMomentum.y() * combinedEnergy / oldFourMomentum.t(),
                                          oldFourMomentum.z() * combinedEnergy / oldFourMomentum.t(),
                                          combinedEnergy);

  ele.correctMomentum(newFourMomentum, ele.trackMomentumError(), combinedEnergyError);
}

void EGRegressionModifierV2::modifyObject(reco::Photon& pho) const {
  // regression calculation needs no additional valuemaps

  const reco::SuperClusterRef& superClus = pho.superCluster();
  const edm::Ptr<reco::CaloCluster>& seed = superClus->seed();

  // skip HGCAL for now
  if (EcalTools::isHGCalDet(seed->seed().det()))
    return;

  const int numberOfClusters = superClus->clusters().size();
  const bool missing_clusters = !superClus->clusters()[numberOfClusters - 1].isAvailable();
  if (missing_clusters)
    return;  // do not apply corrections in case of missing info (slimmed MiniAOD electrons)

  const bool isEB = pho.isEB();

  std::array<float, 32> eval;
  const double rawEnergy = superClus->rawEnergy();
  const double raw_es_energy = superClus->preshowerEnergy();
  const auto& full5x5_pss = pho.full5x5_showerShapeVariables();

  float e5x5Inverse = full5x5_pss.e5x5 != 0. ? vdt::fast_inv(full5x5_pss.e5x5) : 0.;

  eval[0] = rawEnergy;
  eval[1] = superClus->etaWidth();
  eval[2] = superClus->phiWidth();
  eval[3] = superClus->seed()->energy() / rawEnergy;
  eval[4] = full5x5_pss.e5x5 / rawEnergy;
  eval[5] = pho.hadronicOverEm();
  eval[6] = rhoValue_;
  eval[7] = seed->eta() - superClus->position().Eta();
  eval[8] = reco::deltaPhi(seed->phi(), superClus->position().Phi());
  eval[9] = pho.full5x5_r9();
  eval[10] = full5x5_pss.sigmaIetaIeta;
  eval[11] = full5x5_pss.sigmaIetaIphi;
  eval[12] = full5x5_pss.sigmaIphiIphi;
  eval[13] = full5x5_pss.maxEnergyXtal * e5x5Inverse;
  eval[14] = full5x5_pss.e2nd * e5x5Inverse;
  eval[15] = full5x5_pss.eTop * e5x5Inverse;
  eval[16] = full5x5_pss.eBottom * e5x5Inverse;
  eval[17] = full5x5_pss.eLeft * e5x5Inverse;
  eval[18] = full5x5_pss.eRight * e5x5Inverse;
  eval[19] = full5x5_pss.e2x5Max * e5x5Inverse;
  eval[20] = full5x5_pss.e2x5Left * e5x5Inverse;
  eval[21] = full5x5_pss.e2x5Right * e5x5Inverse;
  eval[22] = full5x5_pss.e2x5Top * e5x5Inverse;
  eval[23] = full5x5_pss.e2x5Bottom * e5x5Inverse;
  eval[24] = pho.nSaturatedXtals();
  eval[25] = std::max(0, numberOfClusters);

  // calculate coordinate variables

  if (isEB) {
    float dummy;
    int ieta;
    int iphi;
    egammaTools::localEcalClusterCoordsEB(*seed, *caloGeometry_, dummy, dummy, ieta, iphi, dummy, dummy);
    eval[26] = ieta;
    eval[27] = iphi;
    int signieta = ieta > 0 ? +1 : -1;
    eval[28] = (ieta - signieta) % 5;
    eval[29] = (iphi - 1) % 2;
    eval[30] = (abs(ieta) <= 25) * ((ieta - signieta)) + (abs(ieta) > 25) * ((ieta - 26 * signieta) % 20);
    eval[31] = (iphi - 1) % 20;

  } else {
    float dummy;
    int ix;
    int iy;
    egammaTools::localEcalClusterCoordsEE(*seed, *caloGeometry_, dummy, dummy, ix, iy, dummy, dummy);
    eval[26] = ix;
    eval[27] = iy;
    eval[28] = raw_es_energy / rawEnergy;
  }

  //magic numbers for MINUIT-like transformation of BDT output onto limited range
  //(These should be stored inside the conditions object in the future as well)
  constexpr double meanlimlow = -1.0;
  constexpr double meanlimhigh = 3.0;
  constexpr double meanoffset = meanlimlow + 0.5 * (meanlimhigh - meanlimlow);
  constexpr double meanscale = 0.5 * (meanlimhigh - meanlimlow);

  constexpr double sigmalimlow = 0.0002;
  constexpr double sigmalimhigh = 0.5;
  constexpr double sigmaoffset = sigmalimlow + 0.5 * (sigmalimhigh - sigmalimlow);
  constexpr double sigmascale = 0.5 * (sigmalimhigh - sigmalimlow);

  size_t coridx = 0;
  float rawPt = rawEnergy * superClus->position().rho() / superClus->position().r();
  bool isSaturated = pho.nSaturatedXtals();

  if (rawPt >= lowEnergyEcalOnlyThr_ || (isSaturated && forceHighEnergyEcalTrainingIfSaturated_)) {
    if (isEB)
      coridx = 1;
    else
      coridx = 3;
  } else {
    if (isEB)
      coridx = 0;
    else
      coridx = 2;
  }

  //these are the actual BDT responses
  double rawmean = phoForestsMean_[coridx]->GetResponse(eval.data());
  double rawsigma = phoForestsSigma_[coridx]->GetResponse(eval.data());

  //apply transformation to limited output range (matching the training)
  double mean = meanoffset + meanscale * vdt::fast_sin(rawmean);
  double sigma = sigmaoffset + sigmascale * vdt::fast_sin(rawsigma);

  // Correct the energy. A negative energy means that the correction went
  // outside the boundaries of the training. In this case uses raw.
  // The resolution estimation, on the other hand should be ok.
  if (mean < 0.)
    mean = 1.0;

  const double ecor = mean * (rawEnergy + raw_es_energy);
  const double sigmacor = sigma * ecor;

  pho.setCorrectedEnergy(reco::Photon::P4type::regression2, ecor, sigmacor, true);
}
