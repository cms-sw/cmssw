
#include "CalibMuon/CSCCalibration/interface/CSCFakeDBPedestals.h"

void CSCFakeDBPedestals::prefillDBPedestals()
{
  const int MAX_SIZE = 217728;
  const int PED_FACTOR=10;
  const int RMS_FACTOR=1000;
  cndbpedestals = new CSCDBPedestals();
  cndbpedestals->pedestals.resize(MAX_SIZE);

  seed = 10000;	
  srand(seed);
  meanped=600.0, meanrms=1.5, M=1000;
   
  for(int i=0; i<MAX_SIZE;i++){
    cndbpedestals->pedestals[i].ped=int (((double)rand()/((double)(RAND_MAX)+(double)(1)))*100+meanped*PED_FACTOR+0.5);
    cndbpedestals->pedestals[i].rms=int (((double)rand()/((double)(RAND_MAX)+(double)(1)))+meanrms*RMS_FACTOR+0.5);
  }
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
