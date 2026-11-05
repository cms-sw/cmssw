// -*- C++ -*-
//
// Package:    CalibCalorimetry/HcalTPGEventSetup
// Class:      HcalTPGCoderULUT
//
/**\class HcalTPGCoderULUT HcalTPGCoderULUT.cc src/HcalTPGCoderULUT.cc

 Description: Manages the HcaluLUTTPGCoder and updates on record change

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeremiah Mans
//         Created:  Fri Sep 15 11:49:44 CDT 2006
//

#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/HcaluLUTTPGCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGRecord.h"
#include "CondFormats/DataRecord/interface/HcalTimeSlewRecord.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>

class HcalTPGCoderULUT : public edm::ESProducer {
public:
  HcalTPGCoderULUT(const edm::ParameterSet&);
  ~HcalTPGCoderULUT() override;

  typedef std::shared_ptr<HcalTPGCoder> ReturnType;

  ReturnType produce(const HcalTPGRecord&);

private:
  using HostType = edm::ESProductHost<HcaluLUTTPGCoder, HcalDbRecord>;

  void buildCoder(const HcalTopology*, const HcalElectronicsMap*, const HcalTimeSlew*, HcaluLUTTPGCoder*);

  edm::ReusableObjectHolder<HostType> holder_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> topoToken_;
  edm::ESGetToken<HcalTimeSlew, HcalTimeSlewRecord> delayToken_;
  edm::ESGetToken<HcalDbService, HcalDbRecord> serviceToken_;
  bool LUTGenerationMode_, linearLUTs_;
  bool contain1TSHB_, contain1TSHE_;
  double containPhaseNSHB_, containPhaseNSHE_;
  bool applyFixPCC_;
  bool overrideDBweightsAndFilterHB_, overrideDBweightsAndFilterHE_;
  double nPedWidthsForZS_;
  bool overrideDBnPedWidthsForZS_;
  double linearLSB_QIE8_, linearLSB_QIE11Overlap_, linearLSB_QIE11_;
  int maskBit_;
  bool overrideFGHF_;
  std::array<uint32_t, 2> FG_HF_thresholds_;
  bool overrideHBLLP_;
  std::array<uint32_t, 4> HB_LLP_thresholds_;
};

HcalTPGCoderULUT::HcalTPGCoderULUT(const edm::ParameterSet& iConfig) {
  contain1TSHB_ = iConfig.getParameter<bool>("contain1TSHB");
  contain1TSHE_ = iConfig.getParameter<bool>("contain1TSHE");
  containPhaseNSHB_ = iConfig.getParameter<double>("containPhaseNSHB");
  containPhaseNSHE_ = iConfig.getParameter<double>("containPhaseNSHE");
  overrideDBweightsAndFilterHB_ = iConfig.getParameter<bool>("overrideDBweightsAndFilterHB");
  overrideDBweightsAndFilterHE_ = iConfig.getParameter<bool>("overrideDBweightsAndFilterHE");
  nPedWidthsForZS_ = iConfig.getParameter<double>("nPedWidthsForZS");
  overrideDBnPedWidthsForZS_ = iConfig.getParameter<bool>("overrideDBnPedWidthsForZS");
  applyFixPCC_ = iConfig.getParameter<bool>("applyFixPCC");

  //the following line is needed to tell the framework what
  // data is being produced
  auto cc = setWhatProduced(this);
  topoToken_ = cc.consumes();
  delayToken_ = cc.consumes(edm::ESInputTag{"", "HBHE"});
  serviceToken_ = cc.consumes();

  LUTGenerationMode_ = iConfig.getParameter<bool>("LUTGenerationMode");
  linearLUTs_ = iConfig.getParameter<bool>("linearLUTs");
  auto scales = iConfig.getParameter<edm::ParameterSet>("tpScales").getParameter<edm::ParameterSet>("HBHE");
  linearLSB_QIE8_ = scales.getParameter<double>("LSBQIE8");
  linearLSB_QIE11_ = scales.getParameter<double>("LSBQIE11");
  linearLSB_QIE11Overlap_ = scales.getParameter<double>("LSBQIE11Overlap");
  maskBit_ = iConfig.getParameter<int>("MaskBit");
  overrideFGHF_ = iConfig.getParameter<bool>("overrideFGHF");
  FG_HF_thresholds_ = iConfig.getParameter<std::array<uint32_t, 2> >("FG_HF_thresholds");
  overrideHBLLP_ = iConfig.getParameter<bool>("overrideHBLLP");
  HB_LLP_thresholds_ = iConfig.getParameter<std::array<uint32_t, 4> >("HB_LLP_thresholds");
}

void HcalTPGCoderULUT::buildCoder(const HcalTopology* topo,
                                  const HcalElectronicsMap* emap,
                                  const HcalTimeSlew* delay,
                                  HcaluLUTTPGCoder* theCoder) {
  theCoder->init(topo, emap, delay);

  theCoder->setOverrideDBweightsAndFilterHB(overrideDBweightsAndFilterHB_);
  theCoder->setOverrideDBweightsAndFilterHE(overrideDBweightsAndFilterHE_);

  theCoder->set1TSContainHB(contain1TSHB_);
  theCoder->set1TSContainHE(contain1TSHE_);

  theCoder->setContainPhaseHB(containPhaseNSHB_);
  theCoder->setContainPhaseHE(containPhaseNSHE_);

  theCoder->setNpedWidthsForZS(nPedWidthsForZS_);
  theCoder->setOverrideDBnPedWidthsForZS(overrideDBnPedWidthsForZS_);

  theCoder->setApplyFixPCC(applyFixPCC_);

  theCoder->setAllLinear(linearLUTs_, linearLSB_QIE8_, linearLSB_QIE11_, linearLSB_QIE11Overlap_);
  theCoder->setLUTGenerationMode(LUTGenerationMode_);
  theCoder->setMaskBit(maskBit_);
  theCoder->setOverrideFGHF(overrideFGHF_);
  theCoder->setFGHFthresholds(FG_HF_thresholds_);
  theCoder->setOverrideHBLLP(overrideHBLLP_);
  theCoder->setHBLLPthresholds(HB_LLP_thresholds_);
}

HcalTPGCoderULUT::~HcalTPGCoderULUT() {}

HcalTPGCoderULUT::ReturnType HcalTPGCoderULUT::produce(const HcalTPGRecord& iRecord) {
  auto host = holder_.makeOrGet([]() { return new HostType; });

  const auto& topo = iRecord.get(topoToken_);
  const auto& delayRcd = iRecord.getRecord<HcalDbRecord>();
  const auto& dbServ = iRecord.get(serviceToken_);
  const auto* emap = dbServ.getHcalMapping();
  const auto& delay = delayRcd.get(delayToken_);

  host->ifRecordChanges<HcalDbRecord>(iRecord, [this, &topo, emap, &delay, h = host.get()](auto const& rec) {
    buildCoder(&topo, emap, &delay, h);
    h->update(rec.get(serviceToken_));
  });

  return host;
}

DEFINE_FWK_EVENTSETUP_MODULE(HcalTPGCoderULUT);
