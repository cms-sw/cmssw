#include <memory>

#include <fstream>

#include "CondFormats/CSCObjects/interface/CSCChamberMap.h"
#include "CondFormats/DataRecord/interface/CSCChamberMapRcd.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberMapValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCMap1.h"

CSCChamberMapValues::CSCChamberMapValues(const edm::ParameterSet& iConfig) {
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this, &CSCChamberMapValues::produceChamberMap);
  findingRecord<CSCChamberMapRcd>();
  //now do what ever other initialization is needed
}

CSCChamberMapValues::~CSCChamberMapValues() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
CSCChamberMapValues::ReturnType CSCChamberMapValues::produceChamberMap(const CSCChamberMapRcd& iRecord) {
  //need a new object so to not be deleted at exit
  return ReturnType(fillChamberMap());
}

void CSCChamberMapValues::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                         const edm::IOVSyncValue&,
                                         edm::ValidityInterval& oValidity) {
  oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}
