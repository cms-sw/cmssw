#include <memory>

#include <fstream>

#include "CondFormats/CSCObjects/interface/CSCDDUMap.h"
#include "CondFormats/DataRecord/interface/CSCDDUMapRcd.h"
#include "OnlineDB/CSCCondDB/interface/CSCDDUMapValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCMap1.h"

CSCDDUMapValues::CSCDDUMapValues(const edm::ParameterSet& iConfig) {
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this, &CSCDDUMapValues::produceDDUMap);
  findingRecord<CSCDDUMapRcd>();
  //now do what ever other initialization is needed
}

CSCDDUMapValues::~CSCDDUMapValues() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
CSCDDUMapValues::ReturnType CSCDDUMapValues::produceDDUMap(const CSCDDUMapRcd& iRecord) {
  //need a new object so to not be deleted at exit
  return ReturnType(fillDDUMap());
}

void CSCDDUMapValues::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                     const edm::IOVSyncValue&,
                                     edm::ValidityInterval& oValidity) {
  oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}
