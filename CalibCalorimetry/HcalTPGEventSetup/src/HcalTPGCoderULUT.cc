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
// $Id$
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
  edm::FileInPath filename_;
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
HcalTPGCoderULUT::HcalTPGCoderULUT(const edm::ParameterSet& iConfig) :
  filename_(iConfig.getParameter<edm::FileInPath>("filename"))
{
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

}


//
// member functions
//

// ------------ method called to produce the data  ------------
HcalTPGCoderULUT::ReturnType
HcalTPGCoderULUT::produce(const HcalTPGRecord& iRecord)
{
   using namespace edm::es;
   edm::LogInfo("HCAL") << "Using " << filename_.fullPath() << " for HcalTPGCoderULUT initialization";
   std::auto_ptr<HcalTPGCoder> pHcalTPGCoder(new HcaluLUTTPGCoder(filename_.fullPath().c_str()));

   return pHcalTPGCoder ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_SOURCE(HcalTPGCoderULUT)
