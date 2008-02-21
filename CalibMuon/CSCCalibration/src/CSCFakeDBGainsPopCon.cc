#include <memory>
#include "boost/shared_ptr.hpp"

#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
#include "CondFormats/DataRecord/interface/CSCDBGainsRcd.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeDBGainsPopCon.h"
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

void CSCFakeDBGainsPopCon::prefillDBGains()
{
  const int MAX_SIZE=217728;
  const int FACTOR=1000;
  cndbgains = new CSCDBGains();
  cndbgains->gains.resize(MAX_SIZE);

  seed = 10000;	
  srand(seed);
  mean=6.8, min=-10.0, minchi=1.0, M=1000;

  for(int i=0; i<MAX_SIZE;i++){
   cndbgains->gains[i].gain_slope=int ((((double)rand()/((double)(RAND_MAX)+(double)(1)))+mean)*FACTOR+0.5);
  }
}  

CSCFakeDBGainsPopCon::CSCFakeDBGainsPopCon(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  prefillDBGains();
  setWhatProduced(this,&CSCFakeDBGainsPopCon::produceDBGains);
  findingRecord<CSCDBGainsRcd>();
  //now do what ever other initialization is needed
}


CSCFakeDBGainsPopCon::~CSCFakeDBGainsPopCon()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete cndbgains;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCFakeDBGainsPopCon::ReturnType
CSCFakeDBGainsPopCon::produceDBGains(const CSCDBGainsRcd& iRecord)
{
  CSCDBGains* mydata=new CSCDBGains( *cndbgains );
  return mydata;
  //return cndbgains;  
}

 void CSCFakeDBGainsPopCon::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }
