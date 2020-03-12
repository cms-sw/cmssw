#include <fstream>
#include <memory>

#include "CalibMuon/CSCCalibration/interface/CSCPedestalsDBConditions.h"
#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "CondFormats/DataRecord/interface/CSCDBPedestalsRcd.h"

CSCPedestalsDBConditions::CSCPedestalsDBConditions(const edm::ParameterSet &iConfig) {
  // the following line is needed to tell the framework what
  // data is being produced
  // added by Zhen (changed since 1_2_0)
  setWhatProduced(this, &CSCPedestalsDBConditions::produceDBPedestals);
  findingRecord<CSCDBPedestalsRcd>();
  // now do what ever other initialization is needed
}

CSCPedestalsDBConditions::~CSCPedestalsDBConditions() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
CSCPedestalsDBConditions::ReturnType CSCPedestalsDBConditions::produceDBPedestals(const CSCDBPedestalsRcd &iRecord) {
  // need a new object so to not be deleted at exit
  return CSCPedestalsDBConditions::ReturnType(prefillDBPedestals());
}

void CSCPedestalsDBConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
                                              const edm::IOVSyncValue &,
                                              edm::ValidityInterval &oValidity) {
  oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}
