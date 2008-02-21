#include <memory>
#include <fstream>
#include "boost/shared_ptr.hpp"

#include "CondFormats/CSCObjects/interface/CSCDBCrosstalk.h"
#include "CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeDBCrosstalkPopCon.h"

void CSCFakeDBCrosstalkPopCon::prefillDBFakeCrosstalk(){
  const int MAX_SIZE = 217728;
  const int SLOPE_FACTOR=1000000;
  const int INTERCEPT_FACTOR=10000;
  cndbcrosstalk = new CSCDBCrosstalk();
  cndbcrosstalk->crosstalk.resize(MAX_SIZE);

  seed = 10000;	
  srand(seed);
  mean=-0.0009, min=0.035, minchi=1.5, M=1000;
 
  
  for(int i=0; i<MAX_SIZE;++i){
    cndbcrosstalk->crosstalk[i].xtalk_slope_right = int (-((double)rand()/((double)(RAND_MAX)+(double)(1)))/10000+mean)*SLOPE_FACTOR;
    cndbcrosstalk->crosstalk[i].xtalk_intercept_right=int  (((double)rand()/((double)(RAND_MAX)+(double)(1)))/100+min)*INTERCEPT_FACTOR;
    cndbcrosstalk->crosstalk[i].xtalk_slope_left=int (-((double)rand()/((double)(RAND_MAX)+(double)(1)))/10000+mean)*SLOPE_FACTOR;
    cndbcrosstalk->crosstalk[i].xtalk_intercept_left=int (((double)rand()/((double)(RAND_MAX)+(double)(1)))/100+min)*INTERCEPT_FACTOR;
   }
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


