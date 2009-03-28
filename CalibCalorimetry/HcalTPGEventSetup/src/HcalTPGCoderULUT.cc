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
// $Id: HcalTPGCoderULUT.cc,v 1.6 2008/01/30 08:44:17 tulika Exp $
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
  edm::FileInPath *ifilename_;
  ReturnType coder_;  
  HcaluLUTTPGCoder* theCoder;
  bool read_Ascii;
  bool read_XML;
  bool LUTGenerationMode;
  std::string TagName;
  std::string AlgoName;

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
  /*
  ifilename_=0;  
  try {
    ofilename_=new edm::FileInPath(iConfig.getParameter<edm::FileInPath>("outputLUTs"));
  } catch (...) {
    ifilename_=new edm::FileInPath(iConfig.getParameter<edm::FileInPath>("filename"));
    ofilename_=0;
  }
  */
  read_Ascii=false;
  read_Ascii=iConfig.getParameter<bool>("read_Ascii_LUTs");
  read_XML=false;
  read_XML=iConfig.getParameter<bool>("read_XML_LUTs");
  if (read_Ascii || read_XML) ifilename_=new edm::FileInPath(iConfig.getParameter<edm::FileInPath>("inputLUTs"));
  else ifilename_=new edm::FileInPath(iConfig.getParameter<edm::FileInPath>("filename"));

  LUTGenerationMode = iConfig.getParameter<bool>("LUTGenerationMode");
  TagName = iConfig.getParameter<std::string>("TagName");
  AlgoName = iConfig.getParameter<std::string>("AlgoName");

   //the following line is needed to tell the framework what
   // data is being produced
   if (!(read_Ascii || read_XML)) setWhatProduced(this,(dependsOn(&HcalTPGCoderULUT::dbRecordCallback)));
   else setWhatProduced(this);

  
  //now do what ever other initialization is needed
   using namespace edm::es;
   if (read_Ascii || read_XML){
     edm::LogInfo("HCAL") << "Using ASCII/XML LUTs" << ifilename_->fullPath() << " for HcalTPGCoderULUT initialization";
     theCoder=new HcaluLUTTPGCoder(ifilename_->fullPath().c_str(),read_Ascii,read_XML);
     coder_=ReturnType(theCoder);
   } 
   else {
     edm::LogInfo("HCAL") << "Using " << ifilename_->fullPath() << " for HcalTPGCoderULUT initialization";
     theCoder=new HcaluLUTTPGCoder(ifilename_->fullPath().c_str());
	  theCoder->SetLUTGenerationMode(LUTGenerationMode);
	  theCoder->SetLUTInfo(TagName,AlgoName);
     coder_=ReturnType(theCoder);
   }  
}


HcalTPGCoderULUT::~HcalTPGCoderULUT()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

 
  if (ifilename_!=0) delete ifilename_;
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
  theCoder->update(*conditions);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HcalTPGCoderULUT);
