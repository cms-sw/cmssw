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
// $Id: HcalTPGCoderULUT.cc,v 1.12 2012/11/12 20:44:36 dlange Exp $
//
//


// system include files
#include <memory>
#include <string>
#include "boost/shared_ptr.hpp"

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
      ~HcalTPGCoderULUT();
     
      typedef boost::shared_ptr<HcalTPGCoder> ReturnType;
      void dbRecordCallback(const HcalDbRecord&);

      ReturnType produce(const HcalTPGRecord&);
   private:
  void buildCoder(const HcalTopology*);
      // ----------member data ---------------------------
      ReturnType coder_;  
      HcaluLUTTPGCoder* theCoder_;
  bool read_FGLut_, read_Ascii_,read_XML_,LUTGenerationMode_;
  int maskBit_;
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
     maskBit_ = iConfig.getParameter<int>("MaskBit");
   } else {
     ifilename_=iConfig.getParameter<edm::FileInPath>("inputLUTs");
     setWhatProduced(this);
   }

   theCoder_=0;
}

  
void HcalTPGCoderULUT::buildCoder(const HcalTopology* topo) {  
   using namespace edm::es;
   theCoder_ = new HcaluLUTTPGCoder();
   if (read_Ascii_ || read_XML_){
     edm::LogInfo("HCAL") << "Using ASCII/XML LUTs" << ifilename_.fullPath() << " for HcalTPGCoderULUT initialization";
     if (read_Ascii_) {
       theCoder_->update(ifilename_.fullPath().c_str(),*topo);
     } else if (read_XML_) theCoder_->updateXML(ifilename_.fullPath().c_str(),*topo);

      // Read FG LUT and append to most significant bit 11
     if (read_FGLut_) {
       theCoder_->update(fgfile_.fullPath().c_str(), *topo,true);
   } 
   } else {
     theCoder_->setLUTGenerationMode(LUTGenerationMode_);
     theCoder_->setMaskBit(maskBit_);
   }  
   coder_=ReturnType(theCoder_);
}


HcalTPGCoderULUT::~HcalTPGCoderULUT()
{
 
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
  if (theCoder_==0) {
    edm::ESHandle<HcalTopology> htopo;
    iRecord.getRecord<IdealGeometryRecord>().get(htopo);
    const HcalTopology* topo=&(*htopo);
    buildCoder(topo);
  }


  return coder_;
}

void HcalTPGCoderULUT::dbRecordCallback(const HcalDbRecord& theRec) {
   edm::ESHandle<HcalDbService> conditions;
   theRec.get(conditions);
   const HcalTopology* topo=conditions->getTopologyUsed();

   if (theCoder_==0) {
     buildCoder(topo);
   }

   theCoder_->update(*conditions);

   // Temporary update for FG Lut
   // Will be moved to DB
   if (read_FGLut_) theCoder_->update(fgfile_.fullPath().c_str(),*topo,true);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HcalTPGCoderULUT);
