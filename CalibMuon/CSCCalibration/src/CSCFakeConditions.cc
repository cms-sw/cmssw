#include <memory>
#include "boost/shared_ptr.hpp"

//FW include files
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

//CSCObjects
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/CSCObjects/interface/CSCGains.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeConditions.h"
#include "CondFormats/DataRecord/interface/CSCGainsRcd.h"

CSCFakeConditions::CSCFakeConditions(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  gains.prefillMap();
  // added by Zhen (changed since 1_2_0)
  setWhatProduced(this,&CSCFakeConditions::produce);
  //findingRecord<CSCGainsRcd>();
  //now do what ever other initialization is needed
}


CSCFakeConditions::~CSCFakeConditions()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCFakeConditions::ReturnType
CSCFakeConditions::produce(const GainsRcd& iRecord)
{
    gains.prefillMap();
    gains.print();
    // Added by Zhen, need a new object so to not be deleted at exit
    //    std::cout<<"about to copy"<<std::endl;
    CSCGains* mydata=new CSCGains(gains);
    //    std::cout<<"mydata "<<mydata<<std::endl;
    return mydata;
}

 void CSCFakeConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }
