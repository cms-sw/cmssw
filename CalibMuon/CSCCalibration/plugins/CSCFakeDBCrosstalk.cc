#include "CalibMuon/CSCCalibration/interface/CSCCrosstalkDBConditions.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeDBCrosstalk.h"
#include "CondFormats/CSCObjects/interface/CSCDBCrosstalk.h"
#include "CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h"

CSCFakeDBCrosstalk::CSCFakeDBCrosstalk(const edm::ParameterSet &iConfig) {
  // the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this, &CSCFakeDBCrosstalk::produceDBCrosstalk);
  findingRecord<CSCDBCrosstalkRcd>();
}

CSCFakeDBCrosstalk::~CSCFakeDBCrosstalk() {}

// ------------ method called to produce the data  ------------
CSCFakeDBCrosstalk::Pointer CSCFakeDBCrosstalk::produceDBCrosstalk(const CSCDBCrosstalkRcd &iRecord) {
  return CSCFakeDBCrosstalk::Pointer(prefillDBCrosstalk());
}
