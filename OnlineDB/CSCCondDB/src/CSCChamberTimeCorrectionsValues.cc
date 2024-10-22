#include <memory>

#include <fstream>

#include "CondFormats/CSCObjects/interface/CSCChamberTimeCorrections.h"
#include "CondFormats/DataRecord/interface/CSCChamberTimeCorrectionsRcd.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberTimeCorrectionsValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCCableRead.h"

CSCChamberTimeCorrectionsValues::CSCChamberTimeCorrectionsValues(const edm::ParameterSet& iConfig) {
  //the following line is needed to tell the framework what
  // data is being produced
  isForMC = iConfig.getUntrackedParameter<bool>("isForMC", true);
  ME11offsetMC = 184;
  ME11offsetData = 205;
  nonME11offsetMC = 174;
  nonME11offsetData = 216;
  setWhatProduced(this, &CSCChamberTimeCorrectionsValues::produceChamberTimeCorrections);
  findingRecord<CSCChamberTimeCorrectionsRcd>();
  //now do what ever other initialization is needed
}

CSCChamberTimeCorrectionsValues::~CSCChamberTimeCorrectionsValues() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
CSCChamberTimeCorrectionsValues::ReturnType CSCChamberTimeCorrectionsValues::produceChamberTimeCorrections(
    const CSCChamberTimeCorrectionsRcd& iRecord) {
  //need a new object so to not be deleted at exit
  return ReturnType(
      prefill(isForMC, isForMC ? ME11offsetMC : ME11offsetData, isForMC ? nonME11offsetMC : nonME11offsetData));
}

void CSCChamberTimeCorrectionsValues::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                                     const edm::IOVSyncValue&,
                                                     edm::ValidityInterval& oValidity) {
  oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}
