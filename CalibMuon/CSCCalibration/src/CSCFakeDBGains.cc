#include <memory>
#include "boost/shared_ptr.hpp"

#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
#include "CondFormats/DataRecord/interface/CSCDBGainsRcd.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeDBGains.h"
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/MuonDetId/interface/CSCIndexer.h>

void CSCFakeDBGains::prefillDBGains()
{
  cndbgains = new CSCDBGains();
  const CSCDetId& detId = CSCDetId();
  CSCIndexer* theIndexer = new CSCIndexer;
  
  int max_istrip,id_layer,stripIndex,max_ring,max_cham;
  unsigned layerId;

  seed = 10000;	
  srand(seed);
  mean=6.8, min=-10.0, minchi=1.0, M=1000;

  unsigned int channelCount = 1;

 //endcap=1 to 2,station=1 to 4, ring=1 to 4,chamber=1 to 36,layer=1 to 6 
  for(int iendcap=detId.minEndcapId(); iendcap<=detId.maxEndcapId(); iendcap++){
    for(int istation=detId.minStationId() ; istation<=detId.maxStationId(); istation++){
      max_ring=detId.maxRingId();

      //station 4 ring 4 not there(36 chambers*2 missing)
      //3 rings max this way of counting (ME1a & b)
      if(istation==1)    max_ring=3;
      if(istation==2)    max_ring=2;
      if(istation==3)    max_ring=2;
      if(istation==4)    max_ring=1;
      
      for(int iring=detId.minRingId(); iring<=max_ring; iring++){
	max_istrip=80;
	max_cham=detId.maxChamberId();  
	if(istation==1 && iring==1)    max_cham=36;
	if(istation==1 && iring==2)    max_cham=36;
	if(istation==1 && iring==3)    max_cham=36;
	if(istation==2 && iring==1)    max_cham=18;
	if(istation==2 && iring==2)    max_cham=36;
	if(istation==3 && iring==1)    max_cham=18;
	if(istation==3 && iring==2)    max_cham=36;
	if(istation==4 && iring==1)    max_cham=18;
	
	for(int ichamber=detId.minChamberId(); ichamber<=max_cham; ichamber++){
	  for(int ilayer=detId.minLayerId(); ilayer<=detId.maxLayerId(); ilayer++){
	    //station 1 ring 3 has 64 strips per layer instead of 80 
	    if(istation==1 && iring==3)   max_istrip=64;
	    
	    std::vector<CSCDBGains::Item> itemvector;
	    itemvector.resize(max_istrip);
	    
	    for(int istrip=0;istrip<max_istrip;istrip++){
	      itemvector[istrip].gain_slope=((double)rand()/((double)(RAND_MAX)+(double)(1)))+mean;
	      itemvector[istrip].gain_intercept=((double)rand()/((double)(RAND_MAX)+(double)(1)))+min;
	      itemvector[istrip].gain_chi2=((double)rand()/((double)(RAND_MAX)+(double)(1)))+minchi;
	    }
	    std::copy(itemvector.begin(), itemvector.end(), std::back_inserter(cndbgains->gains));
            channelCount += max_istrip;
	  }
	}
      }
    }
  }
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
