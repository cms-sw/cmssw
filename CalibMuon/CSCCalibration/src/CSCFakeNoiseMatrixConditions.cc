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
    //    matrix.print();
    // Added by Zhen, need a new object so to not be deleted at exit
    //    std::cout<<"about to copy"<<std::endl;
    CSCNoiseMatrix* mydata=new CSCNoiseMatrix(matrix.get());
    
    /*
    std::cout <<"=========================DUMP from produce=====================" << std::endl;
    std::map<int,std::vector<CSCNoiseMatrix::Item> >::const_iterator it;
    for( it=mydata->matrix.begin();it!=mydata->matrix.end(); ++it ){
       std::cout<<"layer id found "<<it->first<<std::endl;
       std::vector<CSCNoiseMatrix::Item>::const_iterator matrixit;
       for( matrixit=it->second.begin(); matrixit!=it->second.end(); ++matrixit ){
         std::cout << "  matrix:  " <<matrixit->matrix_elem33 << " elem34: " << matrixit->matrix_elem34
                   << std::endl;
       }
    }
    std::cout <<"=========================END DUMP from produce=====================" << std::endl;
    */
       return mydata;

}

 void CSCFakeNoiseMatrixConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }
