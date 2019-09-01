#include <fstream>
#include <memory>

#include "CalibMuon/CSCCalibration/interface/CSCChipSpeedCorrectionDBConditions.h"
#include "CondFormats/CSCObjects/interface/CSCDBChipSpeedCorrection.h"
#include "CondFormats/DataRecord/interface/CSCDBChipSpeedCorrectionRcd.h"

CSCChipSpeedCorrectionDBConditions::CSCChipSpeedCorrectionDBConditions(const edm::ParameterSet &iConfig) {
  // the following line is needed to tell the framework what
  // data is being produced
  isForMC = iConfig.getUntrackedParameter<bool>("isForMC", true);
  dataCorrFileName = iConfig.getUntrackedParameter<std::string>("dataCorrFileName", "empty.txt");
  dataOffset = 170.;
  // added by Zhen (changed since 1_2_0)
  setWhatProduced(this, &CSCChipSpeedCorrectionDBConditions::produceDBChipSpeedCorrection);
  findingRecord<CSCDBChipSpeedCorrectionRcd>();
  // now do what ever other initialization is needed
}

CSCChipSpeedCorrectionDBConditions::~CSCChipSpeedCorrectionDBConditions() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
CSCChipSpeedCorrectionDBConditions::ReturnType CSCChipSpeedCorrectionDBConditions::produceDBChipSpeedCorrection(
    const CSCDBChipSpeedCorrectionRcd &iRecord) {
  // need a new object so to not be deleted at exit
  return CSCChipSpeedCorrectionDBConditions::ReturnType(
      prefillDBChipSpeedCorrection(isForMC, dataCorrFileName, dataOffset));
}

void CSCChipSpeedCorrectionDBConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
                                                        const edm::IOVSyncValue &,
                                                        edm::ValidityInterval &oValidity) {
  oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}
