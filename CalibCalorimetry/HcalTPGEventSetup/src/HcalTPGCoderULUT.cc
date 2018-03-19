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
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
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
  void dbRecordCallback(const HcalDbRecord&);

  ReturnType produce(const HcalTPGRecord&);
private:
  void buildCoder(const HcalTopology*, const edm::ESHandle<HcalMCParams>&, const edm::ESHandle<HcalRecoParams>&);
  // ----------member data ---------------------------
  ReturnType coder_;  
  HcaluLUTTPGCoder* theCoder_;
  bool read_FGLut_, read_Ascii_,read_XML_,LUTGenerationMode_,linearLUTs_;
  double linearLSB_QIE8_, linearLSB_QIE11Overlap_, linearLSB_QIE11_;
  int maskBit_;
  unsigned int FG_HF_threshold_;
  edm::FileInPath fgfile_,ifilename_;
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
HcalTPGCoderULUT::HcalTPGCoderULUT(const edm::ParameterSet& iConfig) 
{
  read_Ascii_ = iConfig.getParameter<bool>("read_Ascii_LUTs");
  read_XML_ = iConfig.getParameter<bool>("read_XML_LUTs");
  read_FGLut_ = iConfig.getParameter<bool>("read_FG_LUTs"); 
  fgfile_ = iConfig.getParameter<edm::FileInPath>("FGLUTs");
  //the following line is needed to tell the framework what
  // data is being produced
  if (!(read_Ascii_ || read_XML_)) {
    setWhatProduced(this,(dependsOn(&HcalTPGCoderULUT::dbRecordCallback)));
    LUTGenerationMode_ = iConfig.getParameter<bool>("LUTGenerationMode");
    linearLUTs_ = iConfig.getParameter<bool>("linearLUTs");
    auto scales = iConfig.getParameter<edm::ParameterSet>("tpScales").getParameter<edm::ParameterSet>("HBHE");
    linearLSB_QIE8_ = scales.getParameter<double>("LSBQIE8");
    linearLSB_QIE11_ = scales.getParameter<double>("LSBQIE11");
    linearLSB_QIE11Overlap_ = scales.getParameter<double>("LSBQIE11Overlap");
    maskBit_ = iConfig.getParameter<int>("MaskBit");
    FG_HF_threshold_ = iConfig.getParameter<uint32_t>("FG_HF_threshold"); 
  } else {
    ifilename_=iConfig.getParameter<edm::FileInPath>("inputLUTs");
    setWhatProduced(this);
  }

  theCoder_=nullptr;
}

  
void HcalTPGCoderULUT::buildCoder(const HcalTopology* topo, const edm::ESHandle<HcalMCParams>& mcParams, const edm::ESHandle<HcalRecoParams>& recoParams) {
  using namespace edm::es;
  theCoder_ = new HcaluLUTTPGCoder(topo, mcParams, recoParams);
  if (read_Ascii_ || read_XML_){
    edm::LogInfo("HCAL") << "Using ASCII/XML LUTs" << ifilename_.fullPath() << " for HcalTPGCoderULUT initialization";
    if (read_Ascii_) {
      theCoder_->update(ifilename_.fullPath().c_str());
    } else if (read_XML_) {
      theCoder_->updateXML(ifilename_.fullPath().c_str());
    }
    // Read FG LUT and append to most significant bit 11
    if (read_FGLut_) {
      theCoder_->update(fgfile_.fullPath().c_str(), true);
    } 
  } else {
    theCoder_->setAllLinear(linearLUTs_, linearLSB_QIE8_, linearLSB_QIE11_, linearLSB_QIE11Overlap_);
    theCoder_->setLUTGenerationMode(LUTGenerationMode_);
    theCoder_->setMaskBit(maskBit_);
    theCoder_->setFGHFthreshold(FG_HF_threshold_);
  }  
  coder_=ReturnType(theCoder_);
}


HcalTPGCoderULUT::~HcalTPGCoderULUT() {
  
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to produce the data  ------------
HcalTPGCoderULUT::ReturnType
HcalTPGCoderULUT::produce(const HcalTPGRecord& iRecord)
{
  if (theCoder_==nullptr || (read_Ascii_ || read_XML_)) {// !(read_Ascii_ || read_XML_) goes via dbRecordCallback
    edm::ESHandle<HcalTopology> htopo;
    iRecord.getRecord<HcalRecNumberingRecord>().get(htopo);
    const HcalTopology* topo=&(*htopo);

    edm::ESHandle<HcalMCParams> mcParams;
    iRecord.getRecord<HcalMCParamsRcd>().get(mcParams);

    edm::ESHandle<HcalRecoParams> recoParams;
    iRecord.getRecord<HcalRecoParamsRcd>().get(recoParams);

    buildCoder(topo, mcParams, recoParams);
  }
  

  return coder_;
}

void HcalTPGCoderULUT::dbRecordCallback(const HcalDbRecord& theRec) {
  edm::ESHandle<HcalDbService> conditions;
  theRec.get(conditions);
  edm::ESHandle<HcalTopology> htopo;
  theRec.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);

  edm::ESHandle<HcalMCParams> mcParams;
  theRec.getRecord<HcalMCParamsRcd>().get(mcParams);
  edm::ESHandle<HcalRecoParams> recoParams;
  theRec.getRecord<HcalRecoParamsRcd>().get(recoParams);

  buildCoder(topo, mcParams, recoParams);

  theCoder_->update(*conditions);

  // Temporary update for FG Lut
  // Will be moved to DB
  if (read_FGLut_) theCoder_->update(fgfile_.fullPath().c_str(),true);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HcalTPGCoderULUT);
