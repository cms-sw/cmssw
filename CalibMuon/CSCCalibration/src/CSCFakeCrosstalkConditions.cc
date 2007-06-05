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
#include "CondFormats/CSCObjects/interface/CSCcrosstalk.h"
#include "CalibMuon/CSCCalibration/interface/CSCFakeCrosstalkConditions.h"
#include "CondFormats/DataRecord/interface/CSCcrosstalkRcd.h"

void CSCFakeCrosstalkMap::prefillCrosstalkMap(){
  
  const CSCDetId& detId = CSCDetId();
  cncrosstalk = new CSCcrosstalk();
  
  int max_istrip,id_layer,max_ring,max_cham;

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
	    
	    std::vector<CSCcrosstalk::Item> itemvector;
	    itemvector.resize(max_istrip);
	    id_layer = 100000*iendcap + 10000*istation + 1000*iring + 10*ichamber + ilayer;
	    
	    for(int istrip=0;istrip<max_istrip;istrip++){		  
	      itemvector[istrip].xtalk_slope_right    = -0.001;
	      itemvector[istrip].xtalk_intercept_right= 0.045;
	      itemvector[istrip].xtalk_chi2_right     = 2.00;
	      itemvector[istrip].xtalk_slope_left     = -0.001;
	      itemvector[istrip].xtalk_intercept_left = 0.045;
	      itemvector[istrip].xtalk_chi2_left      = 2.00;
		
	      id_layer = 100000*iendcap+10000*istation+1000*iring+100*ichamber+10*ilayer+ilayer;
	      cncrosstalk->crosstalk[id_layer]=itemvector;
	    }
	  }
	}
      }
    }
  }
} 

CSCFakeCrosstalkConditions::CSCFakeCrosstalkConditions(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  crosstalk.prefillCrosstalkMap();
  // added by Zhen (changed since 1_2_0)
  setWhatProduced(this,&CSCFakeCrosstalkConditions::produceCrosstalk);
  findingRecord<CSCcrosstalkRcd>();
  //now do what ever other initialization is needed
}


CSCFakeCrosstalkConditions::~CSCFakeCrosstalkConditions()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCFakeCrosstalkConditions::ReturnType
CSCFakeCrosstalkConditions::produceCrosstalk(const CSCcrosstalkRcd& iRecord)
{
    crosstalk.prefillCrosstalkMap();
    // Added by Zhen, need a new object so to not be deleted at exit
    CSCcrosstalk* mydata=new CSCcrosstalk(crosstalk.get());
    
    return mydata;

}

 void CSCFakeCrosstalkConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }
