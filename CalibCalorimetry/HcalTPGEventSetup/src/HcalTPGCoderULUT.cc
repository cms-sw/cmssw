// -*- C++ -*-
//
// Package:    HcalTPGCoderULUT
// Class:      HcalTPGCoderULUT
//
/**\class HcalTPGCoderULUT HcalTPGCoderULUT.h src/HcalTPGCoderULUT/interface/HcalTPGCoderULUT.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeremiah Mans
//         Created:  Fri Sep 15 11:49:44 CDT 2006
//
//

// system include files
#include <memory>
#include <string>

// user include files

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"

#include "CalibCalorimetry/HcalTPGAlgos/interface/HcaluLUTTPGCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"

//
// class decleration
//

class HcalTPGCoderULUT : public edm::ESProducer {
public:
  HcalTPGCoderULUT(const edm::ParameterSet&);
  ~HcalTPGCoderULUT() override;

  typedef std::shared_ptr<HcalTPGCoder> ReturnType;

  ReturnType produce(const HcalTPGRecord&);

private:
  using HostType = edm::ESProductHost<HcaluLUTTPGCoder, HcalDbRecord>;

  void buildCoder(const HcalTopology*, const HcalTimeSlew*, HcaluLUTTPGCoder*);

  // ----------member data ---------------------------
  edm::ReusableObjectHolder<HostType> holder_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> topoToken_;
  edm::ESGetToken<HcalTimeSlew, HcalTimeSlewRecord> delayToken_;
  edm::ESGetToken<HcalDbService, HcalDbRecord> serviceToken_;
  bool read_FGLut_, read_Ascii_, read_XML_, LUTGenerationMode_, linearLUTs_;
  bool contain1TSHB_, contain1TSHE_;
  double containPhaseNSHB_, containPhaseNSHE_;
  double linearLSB_QIE8_, linearLSB_QIE11Overlap_, linearLSB_QIE11_;
  int maskBit_;
  std::vector<uint32_t> FG_HF_thresholds_;
  edm::FileInPath fgfile_, ifilename_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HcalTPGCoderULUT::HcalTPGCoderULUT(const edm::ParameterSet& iConfig) {
  read_Ascii_ = iConfig.getParameter<bool>("read_Ascii_LUTs");
  read_XML_ = iConfig.getParameter<bool>("read_XML_LUTs");
  read_FGLut_ = iConfig.getParameter<bool>("read_FG_LUTs");
  fgfile_ = iConfig.getParameter<edm::FileInPath>("FGLUTs");
  contain1TSHB_ = iConfig.getParameter<bool>("contain1TSHB");
  contain1TSHE_ = iConfig.getParameter<bool>("contain1TSHE");
  containPhaseNSHB_ = iConfig.getParameter<double>("containPhaseNSHB");
  containPhaseNSHE_ = iConfig.getParameter<double>("containPhaseNSHE");

  //the following line is needed to tell the framework what
  // data is being produced
  auto cc = setWhatProduced(this);
  topoToken_ = cc.consumes();
  delayToken_ = cc.consumes(edm::ESInputTag{"", "HBHE"});

  if (!(read_Ascii_ || read_XML_)) {
    LUTGenerationMode_ = iConfig.getParameter<bool>("LUTGenerationMode");
    linearLUTs_ = iConfig.getParameter<bool>("linearLUTs");
    auto scales = iConfig.getParameter<edm::ParameterSet>("tpScales").getParameter<edm::ParameterSet>("HBHE");
    linearLSB_QIE8_ = scales.getParameter<double>("LSBQIE8");
    linearLSB_QIE11_ = scales.getParameter<double>("LSBQIE11");
    linearLSB_QIE11Overlap_ = scales.getParameter<double>("LSBQIE11Overlap");
    maskBit_ = iConfig.getParameter<int>("MaskBit");
    FG_HF_thresholds_ = iConfig.getParameter<std::vector<uint32_t> >("FG_HF_thresholds");
    serviceToken_ = cc.consumes();
  } else {
    ifilename_ = iConfig.getParameter<edm::FileInPath>("inputLUTs");
  }
}

void HcalTPGCoderULUT::buildCoder(const HcalTopology* topo, const HcalTimeSlew* delay, HcaluLUTTPGCoder* theCoder) {
  using namespace edm::es;
  theCoder->init(topo, delay);
  theCoder->set1TSContainHB(contain1TSHB_);
  theCoder->set1TSContainHE(contain1TSHE_);
  theCoder->setContainPhaseHB(containPhaseNSHB_);
  theCoder->setContainPhaseHE(containPhaseNSHE_);

  if (read_Ascii_ || read_XML_) {
    edm::LogInfo("HCAL") << "Using ASCII/XML LUTs" << ifilename_.fullPath() << " for HcalTPGCoderULUT initialization";
    if (read_Ascii_) {
      theCoder->update(ifilename_.fullPath().c_str());
    } else if (read_XML_) {
      theCoder->updateXML(ifilename_.fullPath().c_str());
    }
    // Read FG LUT and append to most significant bit 11
    if (read_FGLut_) {
      theCoder->update(fgfile_.fullPath().c_str(), true);
    }
  } else {
    theCoder->setAllLinear(linearLUTs_, linearLSB_QIE8_, linearLSB_QIE11_, linearLSB_QIE11Overlap_);
    theCoder->setLUTGenerationMode(LUTGenerationMode_);
    theCoder->setMaskBit(maskBit_);
    theCoder->setFGHFthresholds(FG_HF_thresholds_);
  }
}

HcalTPGCoderULUT::~HcalTPGCoderULUT() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
HcalTPGCoderULUT::ReturnType HcalTPGCoderULUT::produce(const HcalTPGRecord& iRecord) {
  auto host = holder_.makeOrGet([]() { return new HostType; });

  const auto& topo = iRecord.get(topoToken_);
  const auto& delay = iRecord.getRecord<HcalDbRecord>().get(delayToken_);
  if (read_Ascii_ || read_XML_) {
    buildCoder(&topo, &delay, host.get());
  } else {
    host->ifRecordChanges<HcalDbRecord>(iRecord, [this, &topo, &delay, h = host.get()](auto const& rec) {
      buildCoder(&topo, &delay, h);
      h->update(rec.get(serviceToken_));
      // Temporary update for FG Lut
      // Will be moved to DB
      if (read_FGLut_)
        h->update(fgfile_.fullPath().c_str(), true);
    });
  }
  return host;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HcalTPGCoderULUT);
