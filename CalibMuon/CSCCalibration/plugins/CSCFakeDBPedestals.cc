#include "CalibMuon/CSCCalibration/interface/CSCFakeDBPedestals.h"
#include "CalibMuon/CSCCalibration/interface/CSCPedestalsDBConditions.h"
#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "CondFormats/DataRecord/interface/CSCDBPedestalsRcd.h"

CSCFakeDBPedestals::CSCFakeDBPedestals(const edm::ParameterSet &iConfig) {
  // the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this, &CSCFakeDBPedestals::produceDBPedestals);
  findingRecord<CSCDBPedestalsRcd>();
}

CSCFakeDBPedestals::~CSCFakeDBPedestals() {}

// ------------ method called to produce the data  ------------
CSCFakeDBPedestals::Pointer CSCFakeDBPedestals::produceDBPedestals(const CSCDBPedestalsRcd &iRecord) {
  Pointer cndbPedestals(prefillDBPedestals());
  return cndbPedestals;
}

void CSCFakeDBPedestals::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
                                        const edm::IOVSyncValue &,
                                        edm::ValidityInterval &oValidity) {
  oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}
