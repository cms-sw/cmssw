#include <memory>
#include <fstream>
#include "boost/shared_ptr.hpp"

#include "CondFormats/CSCObjects/interface/CSCDBCrosstalk.h"
#include "CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeDBCrosstalkPopCon.h"

void CSCFakeDBCrosstalkPopCon::prefillDBFakeCrosstalk(){
  
  cndbcrosstalk = new CSCDBCrosstalk();
  
  seed = 10000;	
  srand(seed);
  mean=-0.0009, min=0.035, minchi=1.5, M=1000;
 
  std::vector<CSCDBCrosstalk::Item> *itemvector;
  itemvector = new std::vector<CSCDBCrosstalk::Item> ;
  itemvector->resize(252288);
  
  for(int i=0; i<252288;++i){
    itemvector->at(i).xtalk_slope_right = -((double)rand()/((double)(RAND_MAX)+(double)(1)))/10000+mean;
    itemvector->at(i).xtalk_intercept_right= ((double)rand()/((double)(RAND_MAX)+(double)(1)))/100+min;
    itemvector->at(i).xtalk_chi2_right= ((double)rand()/((double)(RAND_MAX)+(double)(1)))+minchi;
    itemvector->at(i).xtalk_slope_left=-((double)rand()/((double)(RAND_MAX)+(double)(1)))/10000+mean;
    itemvector->at(i).xtalk_intercept_left=((double)rand()/((double)(RAND_MAX)+(double)(1)))/100+min;
    itemvector->at(i).xtalk_chi2_left=((double)rand()/((double)(RAND_MAX)+(double)(1)))+minchi;
  }
  std::copy(itemvector->begin(), itemvector->end(), std::back_inserter(cndbcrosstalk->crosstalk)); 
  delete itemvector;
}


CSCFakeDBCrosstalkPopCon::CSCFakeDBCrosstalkPopCon(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  prefillDBFakeCrosstalk();
  setWhatProduced(this,&CSCFakeDBCrosstalkPopCon::produceDBCrosstalk);
  findingRecord<CSCDBCrosstalkRcd>();
  //now do what ever other initialization is needed
}


CSCFakeDBCrosstalkPopCon::~CSCFakeDBCrosstalkPopCon()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete cndbcrosstalk;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCFakeDBCrosstalkPopCon::ReturnType
CSCFakeDBCrosstalkPopCon::produceDBCrosstalk(const CSCDBCrosstalkRcd& iRecord)
{
  CSCDBCrosstalk* mydata=new CSCDBCrosstalk( *cndbcrosstalk );
  return mydata;
  //return cndbcrosstalk;  
}

 void CSCFakeDBCrosstalkPopCon::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }


