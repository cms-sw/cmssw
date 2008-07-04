#include "CalibMuon/CSCCalibration/interface/CSCFakeDBPedestals.h"
#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "CondFormats/DataRecord/interface/CSCDBPedestalsRcd.h"
#include "CalibMuon/CSCCalibration/interface/CSCPedestalsDBConditions.h"

CSCFakeDBPedestals::CSCFakeDBPedestals(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  cndbPedestals = prefillDBPedestals();
  setWhatProduced(this,&CSCFakeDBPedestals::produceDBPedestals);
  findingRecord<CSCDBPedestalsRcd>();
  //now do what ever other initialization is needed
}


CSCFakeDBPedestals::~CSCFakeDBPedestals()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete cndbPedestals;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCFakeDBPedestals::ReturnType
CSCFakeDBPedestals::produceDBPedestals(const CSCDBPedestalsRcd& iRecord)
{
  CSCDBPedestals* mydata = new CSCDBPedestals( *cndbPedestals);
  return mydata;
}

 void CSCFakeDBPedestals::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }
