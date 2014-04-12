#include <memory>
#include "boost/shared_ptr.hpp"
#include <fstream>

#include "CondFormats/CSCObjects/interface/CSCDBGasGainCorrection.h"
#include "CondFormats/DataRecord/interface/CSCDBGasGainCorrectionRcd.h"
#include "CalibMuon/CSCCalibration/interface/CSCGasGainCorrectionDBConditions.h"


CSCGasGainCorrectionDBConditions::CSCGasGainCorrectionDBConditions(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  isForMC = iConfig.getUntrackedParameter<bool>("isForMC",true);
  dataCorrFileName= iConfig.getUntrackedParameter<std::string>("dataCorrFileName","empty.txt");
  cndbGasGainCorr = prefillDBGasGainCorrection(isForMC,dataCorrFileName);
  // added by Zhen (changed since 1_2_0)
  setWhatProduced(this,&CSCGasGainCorrectionDBConditions::produceDBGasGainCorrection);
  findingRecord<CSCDBGasGainCorrectionRcd>();
  //now do what ever other initialization is needed

}


CSCGasGainCorrectionDBConditions::~CSCGasGainCorrectionDBConditions()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete cndbGasGainCorr;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCGasGainCorrectionDBConditions::ReturnType
CSCGasGainCorrectionDBConditions::produceDBGasGainCorrection(const CSCDBGasGainCorrectionRcd& iRecord)
{
  //need a new object so to not be deleted at exit
  CSCDBGasGainCorrection* mydata=new CSCDBGasGainCorrection( *cndbGasGainCorr );
  return mydata;
  
}

void CSCGasGainCorrectionDBConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
							edm::ValidityInterval & oValidity)
{
  oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
  
}
