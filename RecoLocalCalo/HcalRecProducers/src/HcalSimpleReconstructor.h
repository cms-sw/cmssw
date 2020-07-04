#ifndef HCALSIMPLERECONSTRUCTOR_H
#define HCALSIMPLERECONSTRUCTOR_H 1

#include <memory>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "CondFormats/HcalObjects/interface/HcalRecoParams.h"
#include "CondFormats/HcalObjects/interface/HcalRecoParam.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSimpleRecAlgo.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

namespace edm {
  class ConfigurationDescriptions;
}

/** \class HcalSimpleReconstructor
	
    \author J. Mans - Minnesota
    */
class HcalTopology;
class HcalRecNumberingRecord;
class HcalRecoParamsRcd;
class HcalDbService;
class HcalDbRecord;

class HcalSimpleReconstructor : public edm::stream::EDProducer<> {
public:
  explicit HcalSimpleReconstructor(const edm::ParameterSet& ps);
  ~HcalSimpleReconstructor() override;
  void produce(edm::Event& e, const edm::EventSetup& c) final;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void beginRun(edm::Run const& r, edm::EventSetup const& es) final;
  void endRun(edm::Run const& r, edm::EventSetup const& es) final;

private:
  template <class DIGICOLL, class RECHITCOLL>
  void process(edm::Event& e, const edm::EventSetup& c, const edm::EDGetTokenT<DIGICOLL>& tok);
  HcalSimpleRecAlgo reco_;
  DetId::Detector det_;
  int subdet_;
  HcalOtherSubdetector subdetOther_;
  edm::InputTag inputLabel_;

  edm::EDGetTokenT<HFDigiCollection> tok_hf_;
  edm::EDGetTokenT<HODigiCollection> tok_ho_;
  edm::EDGetTokenT<HcalCalibDigiCollection> tok_calib_;

  bool dropZSmarkedPassed_;  // turn on/off dropping of zero suppression marked and passed digis

  // legacy parameters for config-set values compatibility
  // to be removed after 4_2_0...
  int firstSample_;
  int samplesToAdd_;
  bool tsFromDB_;

  std::unique_ptr<HcalRecoParams> paramTS_;  // firstSample & sampleToAdd from DB

  // ES tokens
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> htopoToken_;
  edm::ESGetToken<HcalRecoParams, HcalRecoParamsRcd> paramsToken_;
  edm::ESGetToken<HcalDbService, HcalDbRecord> conditionsToken_;
};

#endif
