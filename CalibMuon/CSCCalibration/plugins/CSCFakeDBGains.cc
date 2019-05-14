#include "CalibMuon/CSCCalibration/interface/CSCFakeDBGains.h"
#include "CalibMuon/CSCCalibration/interface/CSCGainsDBConditions.h"
#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
#include "CondFormats/DataRecord/interface/CSCDBGainsRcd.h"

CSCFakeDBGains::CSCFakeDBGains(const edm::ParameterSet &iConfig) {
  // the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this, &CSCFakeDBGains::produceDBGains);
  findingRecord<CSCDBGainsRcd>();
}

CSCFakeDBGains::~CSCFakeDBGains() {}

// ------------ method called to produce the data  ------------
CSCFakeDBGains::Pointer CSCFakeDBGains::produceDBGains(const CSCDBGainsRcd &iRecord) {
  return CSCFakeDBGains::Pointer(prefillDBGains());
}

void CSCFakeDBGains::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
                                    const edm::IOVSyncValue &,
                                    edm::ValidityInterval &oValidity) {
  oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}
