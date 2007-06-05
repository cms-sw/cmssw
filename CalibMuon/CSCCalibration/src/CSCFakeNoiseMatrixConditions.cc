#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

//FW include files
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

//CSCObjects
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/CSCObjects/interface/CSCNoiseMatrix.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeNoiseMatrixConditions.h"
#include "CondFormats/DataRecord/interface/CSCNoiseMatrixRcd.h"

CSCFakeNoiseMatrixConditions::CSCFakeNoiseMatrixConditions(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  matrix.prefillNoiseMatrixMap();
  // added by Zhen (changed since 1_2_0)
  setWhatProduced(this,&CSCFakeNoiseMatrixConditions::produceNoiseMatrix);
  findingRecord<CSCNoiseMatrixRcd>();
  //now do what ever other initialization is needed
}


CSCFakeNoiseMatrixConditions::~CSCFakeNoiseMatrixConditions()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCFakeNoiseMatrixConditions::ReturnType
CSCFakeNoiseMatrixConditions::produceNoiseMatrix(const CSCNoiseMatrixRcd& iRecord)
{
    matrix.prefillNoiseMatrixMap();
    // Added by Zhen, need a new object so to not be deleted at exit
    CSCNoiseMatrix* mydata=new CSCNoiseMatrix(matrix.get());
    
    return mydata;

}

 void CSCFakeNoiseMatrixConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }
