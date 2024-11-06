#ifndef ZDCHITRECONSTRUCTOR_RUN3_H
#define ZDCHITRECONSTRUCTOR_RUN3_H 1

#include <memory>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/ZdcSimpleRecAlgo_Run3.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalHFStatusBitFromRecHits.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalHFStatusBitFromDigis.h"
#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalLongRecoParams.h"
#include "CondFormats/HcalObjects/interface/HcalLongRecoParam.h"
#include "CondFormats/HcalObjects/interface/HcalPedestal.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalTimingCorrector.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HBHETimeProfileStatusBitSetter.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HBHETimingShapedFlag.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalADCSaturationFlag.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HFTimingTrustFlag.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

class HcalTopology;
class HcalRecNumberingRecord;
class HcalLongRecoParamsRcd;
class HcalDbService;
class HcalDbRecord;
class HcalChannelQuality;
class HcalChannelQualityRcd;
class HcalSeverityLevelComputer;
class HcalSeverityLevelComputerRcd;

/** \class ZdcHitReconstructor_Run3
	
    \author M. Nickel - Kansas
    ** Based on ZDCHitReconstructor.h by E. Garcia
    */
class ZdcHitReconstructor_Run3 : public edm::stream::EDProducer<> {
public:
  explicit ZdcHitReconstructor_Run3(const edm::ParameterSet& ps);
  ~ZdcHitReconstructor_Run3() override;
  void beginRun(edm::Run const& r, edm::EventSetup const& es) final;
  void endRun(edm::Run const& r, edm::EventSetup const& es) final;
  void produce(edm::Event& e, const edm::EventSetup& c) final;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  ZdcSimpleRecAlgo_Run3 reco_;
  HcalADCSaturationFlag* saturationFlagSetter_;

  DetId::Detector det_;
  int subdet_;
  HcalOtherSubdetector subdetOther_;

  edm::EDGetTokenT<QIE10DigiCollection> tok_input_QIE10;

  int correctionMethodEM_;
  int correctionMethodHAD_;
  int correctionMethodRPD_;
  double ootpuRatioEM_;
  double ootpuRatioHAD_;
  double ootpuRatioRPD_;
  double ootpuFracEM_;
  double ootpuFracHAD_;
  double ootpuFracRPD_;
  std::vector<double> chargeRatiosEM_;
  std::vector<double> chargeRatiosHAD_;
  std::vector<double> chargeRatiosRPD_;
  std::vector<unsigned int> bxTs_;
  int nTs_;
  bool forceSOI_;
  std::vector<unsigned int> signalSOI_;
  std::vector<unsigned int> noiseSOI_;

  bool setSaturationFlags_;  // turn on/off flag indicating ADC saturation
  bool dropZSmarkedPassed_;  // turn on/off dropping of zero suppression marked and passed digis
  bool skipRPD_;             // ignore all channels but EM and HCAL if true
  int maxADCvalue_;          // max adc value for saturation Flag

  std::unique_ptr<HcalLongRecoParams> longRecoParams_;  //noiseTS and signalTS from db

  // ES tokens
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> htopoToken_;
  edm::ESGetToken<HcalLongRecoParams, HcalLongRecoParamsRcd> paramsToken_;
  edm::ESGetToken<HcalDbService, HcalDbRecord> conditionsToken_;
  edm::ESGetToken<HcalChannelQuality, HcalChannelQualityRcd> qualToken_;
  edm::ESGetToken<HcalSeverityLevelComputer, HcalSeverityLevelComputerRcd> sevToken_;
};

#endif
