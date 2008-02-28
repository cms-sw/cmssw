#include "CalibMuon/CSCCalibration/interface/CSCFakeDBCrosstalk.h"
#include "CondFormats/CSCObjects/interface/CSCDBCrosstalk.h"
#include "CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h"
#include "CalibMuon/CSCCalibration/interface/CSCCrosstalkDBConditions.h"

CSCFakeDBCrosstalk::CSCFakeDBCrosstalk(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  cndbCrosstalk = prefillDBCrosstalk();
  setWhatProduced(this,&CSCFakeDBCrosstalk::produceDBCrosstalk);
  findingRecord<CSCDBCrosstalkRcd>();
}


CSCFakeDBCrosstalk::~CSCFakeDBCrosstalk()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete cndbCrosstalk;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCFakeDBCrosstalk::ReturnType
CSCFakeDBCrosstalk::produceDBCrosstalk(const CSCDBCrosstalkRcd& iRecord)
{
  //need a new object so to not be deleted at exit
  CSCDBCrosstalk* mydata=new CSCDBCrosstalk( *cndbCrosstalk );
  return mydata;

}

 void CSCFakeDBCrosstalk::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }


