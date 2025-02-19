#include "CalibMuon/CSCCalibration/interface/CSCFakeDBPedestals.h"
#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "CondFormats/DataRecord/interface/CSCDBPedestalsRcd.h"
#include "CalibMuon/CSCCalibration/interface/CSCPedestalsDBConditions.h"

CSCFakeDBPedestals::CSCFakeDBPedestals(const edm::ParameterSet& iConfig)
{
  cndbPedestals = boost::shared_ptr<CSCDBPedestals> ( prefillDBPedestals() );

  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this,&CSCFakeDBPedestals::produceDBPedestals);
  findingRecord<CSCDBPedestalsRcd>();
}


CSCFakeDBPedestals::~CSCFakeDBPedestals()
{
}

// ------------ method called to produce the data  ------------
CSCFakeDBPedestals::Pointer
CSCFakeDBPedestals::produceDBPedestals(const CSCDBPedestalsRcd& iRecord)
{
  return cndbPedestals;
}

 void CSCFakeDBPedestals::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }
