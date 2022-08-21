/** \class EcalUncalibRecHitRecWeightsAlgo
  *  Template used to compute amplitude, pedestal, time jitter, chi2 of a pulse
  *  using a weights method
  *
  *  \author R. Bruneliere - A. Zabi
  */

#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecWeightsAlgo.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerRunOneDigiBase.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EBShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EEShape.h"

class EcalUncalibRecHitWorkerWeights : public EcalUncalibRecHitWorkerRunOneDigiBase {
public:
  EcalUncalibRecHitWorkerWeights(const edm::ParameterSet&, edm::ConsumesCollector& c);
  EcalUncalibRecHitWorkerWeights() = default;  // for EcalUncalibRecHitFillDescriptionWorkerFactory

  void set(const edm::EventSetup& es) override;
  bool run(const edm::Event& evt,
           const EcalDigiCollection::const_iterator& digi,
           EcalUncalibratedRecHitCollection& result) override;

  edm::ParameterSetDescription getAlgoDescription() override;

protected:
  edm::ESGetToken<EcalPedestals, EcalPedestalsRcd> tokenPeds_;
  edm::ESGetToken<EcalGainRatios, EcalGainRatiosRcd> tokenGains_;
  edm::ESGetToken<EcalWeightXtalGroups, EcalWeightXtalGroupsRcd> tokenGrps_;
  edm::ESGetToken<EcalTBWeights, EcalTBWeightsRcd> tokenWgts_;
  edm::ESHandle<EcalPedestals> peds_;
  edm::ESHandle<EcalGainRatios> gains_;
  edm::ESHandle<EcalWeightXtalGroups> grps_;
  edm::ESHandle<EcalTBWeights> wgts_;

  double pedVec[3];
  double pedRMSVec[3];
  double gainRatios[3];

  const EcalWeightSet::EcalWeightMatrix* weights[2];
  const EcalWeightSet::EcalChi2WeightMatrix* chi2mat[2];

  EcalUncalibRecHitRecWeightsAlgo<EBDataFrame> uncalibMaker_barrel_;
  EcalUncalibRecHitRecWeightsAlgo<EEDataFrame> uncalibMaker_endcap_;

  EEShape testbeamEEShape;  // used in the chi2
  EBShape testbeamEBShape;  // can be replaced by simple shape arrays of floats in the future (kostas)
};

EcalUncalibRecHitWorkerWeights::EcalUncalibRecHitWorkerWeights(const edm::ParameterSet& ps, edm::ConsumesCollector& c)
    : EcalUncalibRecHitWorkerRunOneDigiBase(ps, c),
      tokenPeds_(c.esConsumes<EcalPedestals, EcalPedestalsRcd>()),
      tokenGains_(c.esConsumes<EcalGainRatios, EcalGainRatiosRcd>()),
      tokenGrps_(c.esConsumes<EcalWeightXtalGroups, EcalWeightXtalGroupsRcd>()),
      tokenWgts_(c.esConsumes<EcalTBWeights, EcalTBWeightsRcd>()),
      testbeamEEShape(c),
      testbeamEBShape(c) {}

void EcalUncalibRecHitWorkerWeights::set(const edm::EventSetup& es) {
  gains_ = es.getHandle(tokenGains_);
  peds_ = es.getHandle(tokenPeds_);
  grps_ = es.getHandle(tokenGrps_);
  wgts_ = es.getHandle(tokenWgts_);

  testbeamEEShape.setEventSetup(es);
  testbeamEBShape.setEventSetup(es);
}

bool EcalUncalibRecHitWorkerWeights::run(const edm::Event& evt,
                                         const EcalDigiCollection::const_iterator& itdg,
                                         EcalUncalibratedRecHitCollection& result) {
  DetId detid(itdg->id());

  const EcalPedestals::Item* aped = nullptr;
  const EcalMGPAGainRatio* aGain = nullptr;
  const EcalXtalGroupId* gid = nullptr;
  EcalTBWeights::EcalTDCId tdcid(1);

  if (detid.subdetId() == EcalEndcap) {
    unsigned int hashedIndex = EEDetId(detid).hashedIndex();
    aped = &peds_->endcap(hashedIndex);
    aGain = &gains_->endcap(hashedIndex);
    gid = &grps_->endcap(hashedIndex);
  } else {
    unsigned int hashedIndex = EBDetId(detid).hashedIndex();
    aped = &peds_->barrel(hashedIndex);
    aGain = &gains_->barrel(hashedIndex);
    gid = &grps_->barrel(hashedIndex);
  }

  pedVec[0] = aped->mean_x12;
  pedVec[1] = aped->mean_x6;
  pedVec[2] = aped->mean_x1;
  pedRMSVec[0] = aped->rms_x12;
  pedRMSVec[1] = aped->rms_x6;
  pedRMSVec[2] = aped->rms_x1;
  gainRatios[0] = 1.;
  gainRatios[1] = aGain->gain12Over6();
  gainRatios[2] = aGain->gain6Over1() * aGain->gain12Over6();

  // now lookup the correct weights in the map
  EcalTBWeights::EcalTBWeightMap const& wgtsMap = wgts_->getMap();
  EcalTBWeights::EcalTBWeightMap::const_iterator wit;
  wit = wgtsMap.find(std::make_pair(*gid, tdcid));
  if (wit == wgtsMap.end()) {
    edm::LogError("EcalUncalibRecHitError")
        << "No weights found for EcalGroupId: " << gid->id() << " and  EcalTDCId: " << tdcid
        << "\n  skipping digi with id: " << detid.rawId();

    return false;
  }
  const EcalWeightSet& wset = wit->second;  // this is the EcalWeightSet

  const EcalWeightSet::EcalWeightMatrix& mat1 = wset.getWeightsBeforeGainSwitch();
  const EcalWeightSet::EcalWeightMatrix& mat2 = wset.getWeightsAfterGainSwitch();
  //        const EcalWeightSet::EcalChi2WeightMatrix& mat3 = wset.getChi2WeightsBeforeGainSwitch();
  //        const EcalWeightSet::EcalChi2WeightMatrix& mat4 = wset.getChi2WeightsAfterGainSwitch();

  weights[0] = &mat1;
  weights[1] = &mat2;

  //        chi2mat[0] = &mat3;
  //        chi2mat[1] = &mat4;
  /*
        if (detid.subdetId()==EcalEndcap) {
                result.push_back(uncalibMaker_endcap_.makeRecHit(*itdg, pedVec, gainRatios, weights, chi2mat));
        } else {
                result.push_back(uncalibMaker_barrel_.makeRecHit(*itdg, pedVec, gainRatios, weights, chi2mat));
        }
*/
  if (detid.subdetId() == EcalEndcap) {
    EcalUncalibratedRecHit rhit =
        (uncalibMaker_endcap_.makeRecHit(*itdg, pedVec, pedRMSVec, gainRatios, weights, testbeamEEShape));
    if (((EcalDataFrame)(*itdg)).hasSwitchToGain6())
      rhit.setFlagBit(EcalUncalibratedRecHit::kHasSwitchToGain6);
    if (((EcalDataFrame)(*itdg)).hasSwitchToGain1())
      rhit.setFlagBit(EcalUncalibratedRecHit::kHasSwitchToGain1);
    result.emplace_back(rhit);
  } else {
    EcalUncalibratedRecHit rhit =
        (uncalibMaker_endcap_.makeRecHit(*itdg, pedVec, pedRMSVec, gainRatios, weights, testbeamEBShape));
    if (((EcalDataFrame)(*itdg)).hasSwitchToGain6())
      rhit.setFlagBit(EcalUncalibratedRecHit::kHasSwitchToGain6);
    if (((EcalDataFrame)(*itdg)).hasSwitchToGain1())
      rhit.setFlagBit(EcalUncalibratedRecHit::kHasSwitchToGain1);
    result.emplace_back(rhit);
  }
  return true;
}

edm::ParameterSetDescription EcalUncalibRecHitWorkerWeights::getAlgoDescription() {
  edm::ParameterSetDescription psd;
  return psd;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN(EcalUncalibRecHitWorkerFactory, EcalUncalibRecHitWorkerWeights, "EcalUncalibRecHitWorkerWeights");
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitFillDescriptionWorkerFactory.h"
DEFINE_EDM_PLUGIN(EcalUncalibRecHitFillDescriptionWorkerFactory,
                  EcalUncalibRecHitWorkerWeights,
                  "EcalUncalibRecHitWorkerWeights");
