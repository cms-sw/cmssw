#include <fstream>
#include <memory>

#include "CalibMuon/CSCCalibration/interface/CSCGainsDBConditions.h"
#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
#include "CondFormats/DataRecord/interface/CSCDBGainsRcd.h"

CSCGainsDBConditions::CSCGainsDBConditions(const edm::ParameterSet &iConfig) {
  // the following line is needed to tell the framework what
  // data is being produced
  // added by Zhen (changed since 1_2_0)
  setWhatProduced(this, &CSCGainsDBConditions::produceDBGains);
  findingRecord<CSCDBGainsRcd>();
  // now do what ever other initialization is needed
}

CSCGainsDBConditions::~CSCGainsDBConditions() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
CSCGainsDBConditions::ReturnType CSCGainsDBConditions::produceDBGains(const CSCDBGainsRcd &iRecord) {
  // need a new object so to not be deleted at exit
  return CSCGainsDBConditions::ReturnType(prefillDBGains());
}

void CSCGainsDBConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
                                          const edm::IOVSyncValue &,
                                          edm::ValidityInterval &oValidity) {
  oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}
