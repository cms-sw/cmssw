#include <memory>
#include "boost/shared_ptr.hpp"
#include <fstream>

#include "CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h"
#include "CondFormats/DataRecord/interface/CSCDBNoiseMatrixRcd.h"
#include "CalibMuon/CSCCalibration/interface/CSCNoiseMatrixDBConditions.h"

CSCNoiseMatrixDBConditions::CSCNoiseMatrixDBConditions(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  cndbMatrix = prefillDBNoiseMatrix();
  // added by Zhen (changed since 1_2_0)
  setWhatProduced(this,&CSCNoiseMatrixDBConditions::produceDBNoiseMatrix);
  findingRecord<CSCDBNoiseMatrixRcd>();
  //now do what ever other initialization is needed
}


CSCNoiseMatrixDBConditions::~CSCNoiseMatrixDBConditions()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete cndbMatrix;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCNoiseMatrixDBConditions::ReturnType
CSCNoiseMatrixDBConditions::produceDBNoiseMatrix(const CSCDBNoiseMatrixRcd& iRecord)
{
  //need a new object so to not be deleted at exit
  CSCDBNoiseMatrix* mydata=new CSCDBNoiseMatrix( *cndbMatrix );
  return mydata;
  
}

 void CSCNoiseMatrixDBConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }
