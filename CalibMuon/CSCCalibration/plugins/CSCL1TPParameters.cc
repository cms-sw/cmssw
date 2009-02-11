#include <memory>
#include "boost/shared_ptr.hpp"
#include <fstream>

#include "CondFormats/CSCObjects/interface/CSCL1TPParameters.h"
#include "CondFormats/DataRecord/interface/CSCL1TPParametersRcd.h"
#include "CalibMuon/CSCCalibration/interface/CSCL1TPParametersConditions.h"


CSCL1TPParametersConditions::CSCL1TPParametersConditions(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
   CSCl1TPParameters = prefillCSCL1TPParameters();
  // added by Zhen (changed since 1_2_0)
  setWhatProduced(this,&CSCL1TPParametersConditions::produceCSCL1TPParameters);
  findingRecord<CSCL1TPParametersRcd>();
  //now do what ever other initialization is needed
}


CSCL1TPParametersConditions::~CSCL1TPParametersConditions()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete CSCl1TPParameters;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCL1TPParametersConditions::ReturnType
CSCL1TPParametersConditions::produceCSCL1TPParameters(const CSCL1TPParametersRcd& iRecord)
{
  //need a new object so to not be deleted at exit
  CSCL1TPParameters* mydata=new CSCL1TPParameters( *CSCl1TPParameters);
  return mydata;
  
}

 void CSCL1TPParametersConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }
