#include <memory>
#include "boost/shared_ptr.hpp"

#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
#include "CondFormats/DataRecord/interface/CSCDBGainsRcd.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeDBGains.h"
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
//#include <DataFormats/MuonDetId/interface/CSCIndexer.h>

void CSCFakeDBGains::prefillDBGains()
{
  cndbgains = new CSCDBGains();
 
  seed = 10000;	
  srand(seed);
  mean=6.8, min=-10.0, minchi=1.0, M=1000;

  std::vector<CSCDBGains::Item> *itemvector;
  itemvector = new std::vector<CSCDBGains::Item> ;
  itemvector->resize(217728);
  
  for(int i=0; i<217728;i++){
    itemvector->at(i).gain_slope=((double)rand()/((double)(RAND_MAX)+(double)(1)))+mean;
    itemvector->at(i).gain_intercept=((double)rand()/((double)(RAND_MAX)+(double)(1)))+min;
    itemvector->at(i).gain_chi2=((double)rand()/((double)(RAND_MAX)+(double)(1)))+minchi;
  }
  std::copy(itemvector->begin(), itemvector->end(), std::back_inserter(cndbgains->gains));
  delete itemvector;
}  

CSCFakeDBGains::CSCFakeDBGains(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  prefillDBGains();
  setWhatProduced(this,&CSCFakeDBGains::produceDBGains);
  findingRecord<CSCDBGainsRcd>();
  //now do what ever other initialization is needed
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
