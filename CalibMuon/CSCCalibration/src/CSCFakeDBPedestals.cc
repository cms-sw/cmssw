#include <memory>
#include "boost/shared_ptr.hpp"

#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "CondFormats/DataRecord/interface/CSCDBPedestalsRcd.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeDBPedestals.h"

void CSCFakeDBPedestals::prefillDBPedestals()
{
  cndbpedestals = new CSCDBPedestals();
  
  seed = 10000;	
  srand(seed);
  meanped=600.0, meanrms=1.5, M=1000;
   
  std::vector<CSCDBPedestals::Item> *itemvector;
  itemvector = new std::vector<CSCDBPedestals::Item> ;
  itemvector->resize(217728);
  
  for(int i=0; i<217728;i++){
    itemvector->at(i).ped=((double)rand()/((double)(RAND_MAX)+(double)(1)))*100+meanped;
    itemvector->at(i).rms=((double)rand()/((double)(RAND_MAX)+(double)(1)))+meanrms;
  }
  std::copy(itemvector->begin(), itemvector->end(), std::back_inserter(cndbpedestals->pedestals));
  delete itemvector;
}  

CSCFakeDBPedestals::CSCFakeDBPedestals(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  prefillDBPedestals();
  setWhatProduced(this,&CSCFakeDBPedestals::produceDBPedestals);
  findingRecord<CSCDBPedestalsRcd>();
  //now do what ever other initialization is needed
}


CSCFakeDBPedestals::~CSCFakeDBPedestals()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete cndbpedestals;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCFakeDBPedestals::ReturnType
CSCFakeDBPedestals::produceDBPedestals(const CSCDBPedestalsRcd& iRecord)
{
  return cndbpedestals;
}

 void CSCFakeDBPedestals::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }
