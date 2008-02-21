
#include "CalibMuon/CSCCalibration/interface/CSCFakeDBNoiseMatrix.h"

void CSCFakeDBNoiseMatrix::prefillDBNoiseMatrix()
{
  const int MAX_SIZE = 217728;
  const int FACTOR=1000;
  cndbmatrix = new CSCDBNoiseMatrix();
  cndbmatrix->matrix.resize(MAX_SIZE); 
  
  for(int i=0; i<MAX_SIZE;i++){
    cndbmatrix->matrix[i].elem33 = int (10.0*FACTOR+0.5);
    cndbmatrix->matrix[i].elem34 = int (4.0*FACTOR+0.5);
    cndbmatrix->matrix[i].elem44 = int (10.0*FACTOR+0.5);
    cndbmatrix->matrix[i].elem35 = int (3.0*FACTOR+0.5);
    cndbmatrix->matrix[i].elem45 = int (8.0*FACTOR+0.5);
    cndbmatrix->matrix[i].elem55 = int (10.0*FACTOR+0.5);
    cndbmatrix->matrix[i].elem46 = int (2.0*FACTOR+0.5);
    cndbmatrix->matrix[i].elem56 = int (5.0*FACTOR+0.5);
    cndbmatrix->matrix[i].elem66 = int (10.0*FACTOR+0.5);
    cndbmatrix->matrix[i].elem57 = int (3.0*FACTOR+0.5);
    cndbmatrix->matrix[i].elem67 = int (4.0*FACTOR+0.5);
    cndbmatrix->matrix[i].elem77 = int (10.0*FACTOR+0.5);
  }
}

CSCFakeDBNoiseMatrix::CSCFakeDBNoiseMatrix(const edm::ParameterSet& iConfig)
{
  //tell the framework what data is being produced
  prefillDBNoiseMatrix();  
  setWhatProduced(this,&CSCFakeDBNoiseMatrix::produceDBNoiseMatrix);  
  findingRecord<CSCDBNoiseMatrixRcd>();
}


CSCFakeDBNoiseMatrix::~CSCFakeDBNoiseMatrix()
{
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  delete cndbmatrix; // since not made persistent so we still own it.
  //When using this to write to DB comment out the above line!
}

//
// member functions
//

// ------------ method called to produce the data  ------------
CSCFakeDBNoiseMatrix::ReturnType
CSCFakeDBNoiseMatrix::produceDBNoiseMatrix(const CSCDBNoiseMatrixRcd& iRecord)
{
  return cndbmatrix;
}

void CSCFakeDBNoiseMatrix::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
						  edm::ValidityInterval & oValidity)
{
  oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
  
}
