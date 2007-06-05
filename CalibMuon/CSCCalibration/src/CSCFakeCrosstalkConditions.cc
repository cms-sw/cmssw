#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

//FW include files
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

//CSCObjects
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/CSCObjects/interface/CSCcrosstalk.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeCrosstalkConditions.h"
#include "CondFormats/DataRecord/interface/CSCcrosstalkRcd.h"

CSCFakeCrosstalkConditions::CSCFakeCrosstalkConditions(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  crosstalk.prefillCrosstalkMap();
  // added by Zhen (changed since 1_2_0)
  setWhatProduced(this,&CSCFakeCrosstalkConditions::produceCrosstalk);
  findingRecord<CSCcrosstalkRcd>();
  //now do what ever other initialization is needed
}


CSCFakeCrosstalkConditions::~CSCFakeCrosstalkConditions()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCFakeCrosstalkConditions::ReturnType
CSCFakeCrosstalkConditions::produceCrosstalk(const CSCcrosstalkRcd& iRecord)
{
    crosstalk.prefillCrosstalkMap();
    // Added by Zhen, need a new object so to not be deleted at exit
    CSCcrosstalk* mydata=new CSCcrosstalk(crosstalk.get());
    
    return mydata;

}

 void CSCFakeCrosstalkConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }
