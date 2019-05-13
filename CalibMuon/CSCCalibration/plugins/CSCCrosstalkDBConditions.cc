#include <fstream>
#include <memory>

#include "CalibMuon/CSCCalibration/interface/CSCCrosstalkDBConditions.h"
#include "CondFormats/CSCObjects/interface/CSCDBCrosstalk.h"
#include "CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h"

CSCCrosstalkDBConditions::CSCCrosstalkDBConditions(const edm::ParameterSet &iConfig) {
  // the following line is needed to tell the framework what
  // data is being produced
  // added by Zhen (changed since 1_2_0)
  setWhatProduced(this, &CSCCrosstalkDBConditions::produceDBCrosstalk);
  findingRecord<CSCDBCrosstalkRcd>();
  // now do what ever other initialization is needed
}

CSCCrosstalkDBConditions::~CSCCrosstalkDBConditions() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
CSCCrosstalkDBConditions::ReturnType CSCCrosstalkDBConditions::produceDBCrosstalk(const CSCDBCrosstalkRcd &iRecord) {
  // need a new object so to not be deleted at exit
  return CSCCrosstalkDBConditions::ReturnType(prefillDBCrosstalk());
}

void CSCCrosstalkDBConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
                                              const edm::IOVSyncValue &,
                                              edm::ValidityInterval &oValidity) {
  oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}
