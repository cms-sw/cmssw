#include <memory>
#include "boost/shared_ptr.hpp"

#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h"
#include "CondFormats/DataRecord/interface/CSCDBNoiseMatrixRcd.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeDBNoiseMatrixPopCon.h"

void CSCFakeDBNoiseMatrixPopCon::prefillDBNoiseMatrix(){

  cndbmatrix = new CSCDBNoiseMatrix();
  std::vector<CSCDBNoiseMatrix::Item> *itemvector;
  itemvector = new std::vector<CSCDBNoiseMatrix::Item> ;
  itemvector->resize(252288);
  
  for(int i=0; i<252288;i++){
    itemvector->at(i).elem33 = 10.0;
    itemvector->at(i).elem34 = 4.0;
    itemvector->at(i).elem44 = 10.0;
    itemvector->at(i).elem35 = 3.0;
    itemvector->at(i).elem45 = 8.0;
    itemvector->at(i).elem55 = 10.0;
    itemvector->at(i).elem46 = 2.0;
    itemvector->at(i).elem56 = 5.0;
    itemvector->at(i).elem66 = 10.0;
    itemvector->at(i).elem57 = 3.0;
    itemvector->at(i).elem67 = 4.0;
    itemvector->at(i).elem77 = 10.0;
  }
  std::copy(itemvector->begin(), itemvector->end(), std::back_inserter(cndbmatrix->matrix));
  delete itemvector;
}

CSCFakeDBNoiseMatrixPopCon::CSCFakeDBNoiseMatrixPopCon(const edm::ParameterSet& iConfig)
{
  
  //tell the framework what data is being produced
  prefillDBNoiseMatrix();  
  setWhatProduced(this,&CSCFakeDBNoiseMatrixPopCon::produceDBNoiseMatrix);
  
  findingRecord<CSCDBNoiseMatrixRcd>();
  
  //now do what ever other initialization is needed
  
}


CSCFakeDBNoiseMatrixPopCon::~CSCFakeDBNoiseMatrixPopCon()
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
CSCFakeDBNoiseMatrixPopCon::ReturnType
CSCFakeDBNoiseMatrixPopCon::produceDBNoiseMatrix(const CSCDBNoiseMatrixRcd& iRecord)
{
  CSCDBNoiseMatrix* mydata=new CSCDBNoiseMatrix( *cndbmatrix );
  return mydata;
  //return cndbmatrix;
}

void CSCFakeDBNoiseMatrixPopCon::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
						  edm::ValidityInterval & oValidity)
{
  oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
  
}
