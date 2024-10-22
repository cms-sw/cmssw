#ifndef ZDCSIMPLERECONSTRUCTOR_H
#define ZDCSIMPLERECONSTRUCTOR_H 1

#include <memory>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "CondFormats/HcalObjects/interface/HcalLongRecoParams.h"
#include "CondFormats/HcalObjects/interface/HcalLongRecoParam.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/ZdcSimpleRecAlgo.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

class HcalTopology;
class HcalRecNumberingRecord;
class HcalLongRecoParamsRcd;
class HcalDbService;
class HcalDbRecord;
class HcalTimeSlew;
class HcalTimeSlewRecord;

/** \class HcalSimpleReconstructor	
    \author E. Garcia - CSU
    ** Based on HcalSimpleReconstructor.h by J. Mans
    */
class ZdcSimpleReconstructor : public edm::stream::EDProducer<> {
public:
  explicit ZdcSimpleReconstructor(const edm::ParameterSet& ps);
  ~ZdcSimpleReconstructor() override;
  void beginRun(edm::Run const& r, edm::EventSetup const& es) final;
  void endRun(edm::Run const& r, edm::EventSetup const& es) final;
  void produce(edm::Event& e, const edm::EventSetup& c) override;

private:
  ZdcSimpleRecAlgo reco_;
  DetId::Detector det_;
  int subdet_;
  HcalOtherSubdetector subdetOther_;
  edm::EDGetTokenT<ZDCDigiCollection> tok_input_hcal;
  edm::EDGetTokenT<ZDCDigiCollection> tok_input_castor;

  bool dropZSmarkedPassed_;  // turn on/off dropping of zero suppression marked and passed digis

  std::unique_ptr<HcalLongRecoParams> longRecoParams_;  //noiseTS and signalTS from db

  const HcalTimeSlew* hcalTimeSlew_delay_;

  // ES tokens
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> htopoToken_;
  edm::ESGetToken<HcalLongRecoParams, HcalLongRecoParamsRcd> paramsToken_;
  edm::ESGetToken<HcalDbService, HcalDbRecord> conditionsToken_;
  edm::ESGetToken<HcalTimeSlew, HcalTimeSlewRecord> timeSlewToken_;
};

#endif
