#include "CalibMuon/CSCCalibration/interface/CSCFakeDBCrosstalk.h"
#include "CondFormats/CSCObjects/interface/CSCDBCrosstalk.h"
#include "CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h"
#include "CalibMuon/CSCCalibration/interface/CSCCrosstalkDBConditions.h"

CSCFakeDBCrosstalk::CSCFakeDBCrosstalk(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  cndbCrosstalk = boost::shared_ptr<CSCDBCrosstalk> ( prefillDBCrosstalk() );
  setWhatProduced(this,&CSCFakeDBCrosstalk::produceDBCrosstalk);
  findingRecord<CSCDBCrosstalkRcd>();
}


CSCFakeDBCrosstalk::~CSCFakeDBCrosstalk()
{
}


// ------------ method called to produce the data  ------------
CSCFakeDBCrosstalk::Pointer
CSCFakeDBCrosstalk::produceDBCrosstalk(const CSCDBCrosstalkRcd& iRecord)
{
  return cndbCrosstalk;
}

 void CSCFakeDBCrosstalk::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }


