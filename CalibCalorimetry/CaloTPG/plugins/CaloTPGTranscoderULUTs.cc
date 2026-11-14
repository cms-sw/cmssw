// -*- C++ -*-
//
// Package:    CalibCalorimetry/CaloTPG
// Class:      CaloTPGTranscoderULUTs
//
/**\class CaloTPGTranscoderULUTs CaloTPGTranscoderULUTs.cc

 Description: Handles CaloTPGRecords in order to setup and produce a CaloTPGTranscoderULUT

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeremiah Mans
//         Created:  Fri Sep 15 11:49:44 CDT 2006
//
//

#include "CalibCalorimetry/CaloTPG/interface/CaloTPGTranscoderULUT.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CondFormats/DataRecord/interface/HcalLutMetadataRcd.h"
#include "CondFormats/HcalObjects/interface/HcalLutMetadata.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

#include <memory>
#include <vector>

class CaloTPGTranscoderULUTs : public edm::ESProducer {
public:
  CaloTPGTranscoderULUTs(const edm::ParameterSet&);
  ~CaloTPGTranscoderULUTs() override;

  typedef std::unique_ptr<CaloTPGTranscoder> ReturnType;

  ReturnType produce(const CaloTPGRecord&);

private:
  const bool linearLUTs_;
  const double nominal_gain;
  const int NCTScaleShift;
  const int RCTScaleShift;
  const double lsbQIE8;
  const double lsbQIE11;
  edm::ESGetToken<HcalLutMetadata, HcalLutMetadataRcd> lutMetadataToken;
  edm::ESGetToken<HcalTrigTowerGeometry, CaloGeometryRecord> theTrigTowerGeometryToken;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> topoToken;
};

CaloTPGTranscoderULUTs::CaloTPGTranscoderULUTs(const edm::ParameterSet& iConfig)
    : linearLUTs_(iConfig.getParameter<bool>("linearLUTs")),
      nominal_gain(iConfig.getParameter<double>("nominal_gain")),
      NCTScaleShift(iConfig.getParameter<edm::ParameterSet>("tpScales")
                        .getParameter<edm::ParameterSet>("HF")
                        .getParameter<int>("NCTShift")),
      RCTScaleShift(iConfig.getParameter<edm::ParameterSet>("tpScales")
                        .getParameter<edm::ParameterSet>("HF")
                        .getParameter<int>("RCTShift")),
      lsbQIE8(iConfig.getParameter<edm::ParameterSet>("tpScales")
                  .getParameter<edm::ParameterSet>("HBHE")
                  .getParameter<double>("LSBQIE8")),
      lsbQIE11(iConfig.getParameter<edm::ParameterSet>("tpScales")
                   .getParameter<edm::ParameterSet>("HBHE")
                   .getParameter<double>("LSBQIE11")) {
  auto cc = setWhatProduced(this);
  lutMetadataToken = cc.consumes();
  theTrigTowerGeometryToken = cc.consumes();
  topoToken = cc.consumes();
}

CaloTPGTranscoderULUTs::~CaloTPGTranscoderULUTs() {}

CaloTPGTranscoderULUTs::ReturnType CaloTPGTranscoderULUTs::produce(const CaloTPGRecord& iRecord) {
  const auto& lutMetadata = iRecord.get(lutMetadataToken);
  const auto& theTrigTowerGeometry = iRecord.get(theTrigTowerGeometryToken);
  const auto& topoRecord = iRecord.getRecord<HcalLutMetadataRcd>();
  const auto& topo = topoRecord.get(topoToken);

  HcalLutMetadata fullLut{lutMetadata};
  fullLut.setTopo(&topo);

  auto pTCoder = std::make_unique<CaloTPGTranscoderULUT>();
  pTCoder->setup(fullLut, theTrigTowerGeometry, NCTScaleShift, RCTScaleShift, lsbQIE8, lsbQIE11, linearLUTs_);
  return pTCoder;
}

DEFINE_FWK_EVENTSETUP_MODULE(CaloTPGTranscoderULUTs);
