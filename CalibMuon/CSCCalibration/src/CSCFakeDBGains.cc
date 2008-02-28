#include "CalibMuon/CSCCalibration/interface/CSCFakeDBGains.h"
#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
#include "CondFormats/DataRecord/interface/CSCDBGainsRcd.h"
#include "CalibMuon/CSCCalibration/interface/CSCGainsDBConditions.h"

CSCFakeDBGains::CSCFakeDBGains(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  cndbGains = prefillDBGains();
  setWhatProduced(this,&CSCFakeDBGains::produceDBGains);
  findingRecord<CSCDBGainsRcd>();
}


CSCFakeDBGains::~CSCFakeDBGains()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete cndbGains;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCFakeDBGains::ReturnType
CSCFakeDBGains::produceDBGains(const CSCDBGainsRcd& iRecord)
{
  //need a new object so to not be deleted at exit
  CSCDBGains* mydata=new CSCDBGains( *cndbGains );
  return mydata;

}

 void CSCFakeDBGains::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }
