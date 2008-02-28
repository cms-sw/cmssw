#include "CalibMuon/CSCCalibration/interface/CSCFakeDBNoiseMatrix.h"
#include "CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h"
#include "CondFormats/DataRecord/interface/CSCDBNoiseMatrixRcd.h"
#include "CalibMuon/CSCCalibration/interface/CSCNoiseMatrixDBConditions.h"

CSCFakeDBNoiseMatrix::CSCFakeDBNoiseMatrix(const edm::ParameterSet& iConfig)
{
  //tell the framework what data is being produced
  cndbNoiseMatrix = prefillDBNoiseMatrix();  
  setWhatProduced(this,&CSCFakeDBNoiseMatrix::produceDBNoiseMatrix);  
  findingRecord<CSCDBNoiseMatrixRcd>();
}


CSCFakeDBNoiseMatrix::~CSCFakeDBNoiseMatrix()
{
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  delete cndbNoiseMatrix; 
}

//
// member functions
//

// ------------ method called to produce the data  ------------
CSCFakeDBNoiseMatrix::ReturnType
CSCFakeDBNoiseMatrix::produceDBNoiseMatrix(const CSCDBNoiseMatrixRcd& iRecord)
{
  //need a new object so to not be deleted at exit
  CSCDBNoiseMatrix* mydata=new CSCDBNoiseMatrix( *cndbNoiseMatrix );
  return mydata;
}

void CSCFakeDBNoiseMatrix::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
						  edm::ValidityInterval & oValidity)
{
  oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
  
}
