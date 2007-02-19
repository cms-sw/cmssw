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
// $Id: HcalTPGCoderULUT.cc,v 1.2 2006/10/27 01:35:15 wmtan Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/HcaluLUTTPGCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGRecord.h"
#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


//
// class decleration
//

class HcalTPGCoderULUT : public edm::ESProducer,
			 public edm::EventSetupRecordIntervalFinder {
public:
  HcalTPGCoderULUT(const edm::ParameterSet&);
  ~HcalTPGCoderULUT();
  
  typedef std::auto_ptr<HcalTPGCoder> ReturnType;
  
  ReturnType produce(const HcalTPGRecord&);

  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey& iKey, const edm::IOVSyncValue& iTime, edm::ValidityInterval& oInterval ) {
    oInterval = edm::ValidityInterval (edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime()); //infinite
  }
private:
  // ----------member data ---------------------------
  edm::FileInPath *ifilename_,*ofilename_;
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
  ifilename_=0;
  try {
    ofilename_=new edm::FileInPath(iConfig.getParameter<edm::FileInPath>("outputLUTs"));
  } catch (...) {
    ifilename_=new edm::FileInPath(iConfig.getParameter<edm::FileInPath>("filename"));
    ofilename_=0;
  }

  if (ifilename_==0) {
    ifilename_=new edm::FileInPath(iConfig.getParameter<edm::FileInPath>("inputLUTs"));
  }
  
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);
   findingRecord<HcalTPGRecord>();

   //now do what ever other initialization is needed
}


HcalTPGCoderULUT::~HcalTPGCoderULUT()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

  if (ofilename_!=0) delete ofilename_;
  if (ifilename_!=0) delete ifilename_;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
HcalTPGCoderULUT::ReturnType
HcalTPGCoderULUT::produce(const HcalTPGRecord& iRecord)
{
   using namespace edm::es;
   if (ofilename_!=0) {
     edm::LogInfo("HCAL") << "Using " << ifilename_->fullPath() << " and " << ofilename_->fullPath() << " for HcalTPGCoderULUT initialization";
     std::auto_ptr<HcalTPGCoder> pHcalTPGCoder(new HcaluLUTTPGCoder(ifilename_->fullPath().c_str(),ofilename_->fullPath().c_str()));
     return pHcalTPGCoder ;
   } else {
     edm::LogInfo("HCAL") << "Using " << ifilename_->fullPath() << " for HcalTPGCoderULUT initialization";
     std::auto_ptr<HcalTPGCoder> pHcalTPGCoder(new HcaluLUTTPGCoder(ifilename_->fullPath().c_str()));
     return pHcalTPGCoder ;
   }
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_SOURCE(HcalTPGCoderULUT);
