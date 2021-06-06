#ifndef CASTORSIMPLERECONSTRUCTOR_H
#define CASTORSIMPLERECONSTRUCTOR_H 1

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "RecoLocalCalo/CastorReco/interface/CastorSimpleRecAlgo.h"
#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
#include "CalibFormats/CastorObjects/interface/CastorDbRecord.h"
#include "CondFormats/DataRecord/interface/CastorRecoParamsRcd.h"
#include "CondFormats/CastorObjects/interface/CastorRecoParams.h"
#include "CondFormats/DataRecord/interface/CastorSaturationCorrsRcd.h"
#include "CondFormats/CastorObjects/interface/CastorSaturationCorrs.h"

class CastorSimpleReconstructor : public edm::stream::EDProducer<> {
public:
  explicit CastorSimpleReconstructor(const edm::ParameterSet& ps);
  ~CastorSimpleReconstructor() override;
  void produce(edm::Event& e, const edm::EventSetup& c) override;

private:
  CastorSimpleRecAlgo reco_;
  DetId::Detector det_;
  int subdet_;
  //      HcalOtherSubdetector subdetOther_;
  edm::EDGetTokenT<CastorDigiCollection> tok_input_;
  edm::ESGetToken<CastorDbService, CastorDbRecord> tok_conditions_;
  edm::ESGetToken<CastorRecoParams, CastorRecoParamsRcd> tok_recoParams_;
  edm::ESGetToken<CastorSaturationCorrs, CastorSaturationCorrsRcd> tok_satCorr_;

  int firstSample_;
  int samplesToAdd_;
  int maxADCvalue_;
  bool tsFromDB_;
  bool setSaturationFlag_;
  bool doSaturationCorr_;
};

#endif
