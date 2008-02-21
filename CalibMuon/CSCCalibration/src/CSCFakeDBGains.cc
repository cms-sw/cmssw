#include "CalibMuon/CSCCalibration/interface/CSCFakeDBGains.h"

void CSCFakeDBGains::prefillDBGains()
{
  const int MAX_SIZE = 217728;
  const int FACTOR=1000;
  cndbgains = new CSCDBGains();
  cndbgains->gains.resize(MAX_SIZE);

  seed = 10000;	
  srand(seed);
  mean=6.8, min=-10.0, minchi=1.0, M=1000;

  for(int i=0; i<MAX_SIZE;i++){
    cndbgains->gains[i].gain_slope= int (((double)rand()/((double)(RAND_MAX)+(double)(1)))+mean*FACTOR+0.5);
  }
}  

CSCFakeDBGains::CSCFakeDBGains(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  prefillDBGains();
  setWhatProduced(this,&CSCFakeDBGains::produceDBGains);
  findingRecord<CSCDBGainsRcd>();
}


CSCFakeDBGains::~CSCFakeDBGains()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete cndbgains;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCFakeDBGains::ReturnType
CSCFakeDBGains::produceDBGains(const CSCDBGainsRcd& iRecord)
{
  return cndbgains;  
}

 void CSCFakeDBGains::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }
