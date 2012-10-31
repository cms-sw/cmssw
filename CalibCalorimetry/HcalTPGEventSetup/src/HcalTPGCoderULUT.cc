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
// $Id: HcalTPGCoderULUT.cc,v 1.10 2009/10/27 12:06:34 kvtsang Exp $
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
      // ----------member data ---------------------------
      ReturnType coder_;  
      HcaluLUTTPGCoder* theCoder_;
      bool read_FGLut_;
      edm::FileInPath fgfile_;
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
   bool read_Ascii = iConfig.getParameter<bool>("read_Ascii_LUTs");
   bool read_XML = iConfig.getParameter<bool>("read_XML_LUTs");
   read_FGLut_ = iConfig.getParameter<bool>("read_FG_LUTs"); 
   fgfile_ = iConfig.getParameter<edm::FileInPath>("FGLUTs");
   const edm::ParameterSet hcalTopoConsts = iConfig.getParameter<edm::ParameterSet>( "hcalTopologyConstants" );
   StringToEnumParser<HcalTopologyMode::Mode> parser;
   HcalTopologyMode::Mode mode = (HcalTopologyMode::Mode) parser.parseString(hcalTopoConsts.getParameter<std::string>("mode"));
   
   //the following line is needed to tell the framework what
   // data is being produced
   if (!(read_Ascii || read_XML)) setWhatProduced(this,(dependsOn(&HcalTPGCoderULUT::dbRecordCallback)));
   else setWhatProduced(this);
  
  //now do what ever other initialization is needed
   using namespace edm::es;
   theCoder_ = new HcaluLUTTPGCoder();
   if (read_Ascii || read_XML){
      edm::FileInPath ifilename(iConfig.getParameter<edm::FileInPath>("inputLUTs"));
      edm::LogInfo("HCAL") << "Using ASCII/XML LUTs" << ifilename.fullPath() << " for HcalTPGCoderULUT initialization";
      if (read_Ascii) theCoder_->update(ifilename.fullPath().c_str());
      else if (read_XML) theCoder_->updateXML(ifilename.fullPath().c_str(),
					      mode,
					      hcalTopoConsts.getParameter<int>("maxDepthHB"),
					      hcalTopoConsts.getParameter<int>("maxDepthHE"));

      // Read FG LUT and append to most significant bit 11
      if (read_FGLut_) theCoder_->update(fgfile_.fullPath().c_str(), true);
   } 
   else {
      bool LUTGenerationMode = iConfig.getParameter<bool>("LUTGenerationMode");
      int maskBit = iConfig.getParameter<int>("MaskBit");
      theCoder_->setLUTGenerationMode(LUTGenerationMode);
      theCoder_->setMaskBit(maskBit);
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
  return coder_;
}

void HcalTPGCoderULUT::dbRecordCallback(const HcalDbRecord& theRec) {
   edm::ESHandle<HcalDbService> conditions;
   theRec.get(conditions);
   theCoder_->update(*conditions);

   // Temporary update for FG Lut
   // Will be moved to DB
   if (read_FGLut_) theCoder_->update(fgfile_.fullPath().c_str(), true);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HcalTPGCoderULUT);
