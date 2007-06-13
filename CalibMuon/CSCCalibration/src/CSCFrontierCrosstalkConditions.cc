#include <memory>
#include <fstream>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

//FW include files
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

//CSCObjects
//#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "CondFormats/CSCObjects/interface/CSCcrosstalk.h"
#include "CalibMuon/CSCCalibration/interface/CSCFrontierCrosstalkConditions.h"
#include "CondFormats/DataRecord/interface/CSCcrosstalkRcd.h"

void CSCFrontierCrosstalkMap::prefillCrosstalkMap(){
  
  const CSCDetId& detId = CSCDetId();
  cncrosstalk = new CSCcrosstalk();
  
  int max_istrip,id_layer,max_ring,max_cham;
  seed = 10000;	
  srand(seed);
  mean=-0.0009, min=0.035, minchi=1.5, M=1000;

  //endcap=1 to 2,station=1 to 4, ring=1 to 4,chamber=1 to 36,layer=1 to 6 
  std::ifstream indata; 
  indata.open("xtalk.dat"); 
  if(!indata) {
    std::cerr << "Error: xtalk.dat file could not be opened" << std::endl;
    exit(1);
  }
  
  while (!indata.eof() ) { 
    indata>>chamber_id>>slope_right>>intercept_right>>chi2_right>>slope_left>>intercept_left>>chi2_left ; 
    cham_id.push_back(chamber_id);
    slope_r.push_back(slope_right);
    slope_l.push_back(slope_left);
    intercept_r.push_back(intercept_right);
    intercept_l.push_back(intercept_left);
    chi2_r.push_back(chi2_right);
    chi2_l.push_back(chi2_left);
  }

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
	    
	      itemvector[istrip].xtalk_slope_right=-((double)rand()/((double)(RAND_MAX)+(double)(1)))/10000+mean;
	      itemvector[istrip].xtalk_intercept_right=((double)rand()/((double)(RAND_MAX)+(double)(1)))/100+min;
	      itemvector[istrip].xtalk_chi2_right=((double)rand()/((double)(RAND_MAX)+(double)(1)))+minchi;
	      itemvector[istrip].xtalk_slope_left=-((double)rand()/((double)(RAND_MAX)+(double)(1)))/10000+mean;
	      itemvector[istrip].xtalk_intercept_left=((double)rand()/((double)(RAND_MAX)+(double)(1)))/100+min;
	      itemvector[istrip].xtalk_chi2_left=((double)rand()/((double)(RAND_MAX)+(double)(1)))+minchi;
	      cncrosstalk->crosstalk[id_layer]=itemvector;

	      if(istrip==0){
		itemvector[istrip].xtalk_slope_right=-((double)rand()/((double)(RAND_MAX)+(double)(1)))/10000+mean;
		itemvector[istrip].xtalk_intercept_right=((double)rand()/((double)(RAND_MAX)+(double)(1)))/100+min;
		itemvector[istrip].xtalk_chi2_right=((double)rand()/((double)(RAND_MAX)+(double)(1)))+minchi;
		itemvector[istrip].xtalk_slope_left=0.0;
		itemvector[istrip].xtalk_intercept_left=0.0;
		itemvector[istrip].xtalk_chi2_left=0.0;
		cncrosstalk->crosstalk[id_layer]=itemvector;
	      }
	      
	      if(istrip==79){
		itemvector[istrip].xtalk_slope_right=0.0;
		itemvector[istrip].xtalk_intercept_right=0.0;
		itemvector[istrip].xtalk_chi2_right=0.0;
		itemvector[istrip].xtalk_slope_left=-((double)rand()/((double)(RAND_MAX)+(double)(1)))/10000+mean;
		itemvector[istrip].xtalk_intercept_left=((double)rand()/((double)(RAND_MAX)+(double)(1)))/100+min;
		itemvector[istrip].xtalk_chi2_left=((double)rand()/((double)(RAND_MAX)+(double)(1)))+minchi;
		cncrosstalk->crosstalk[id_layer]=itemvector;
	      }
	      
	      for(unsigned int newId=0; newId<cham_id.size();newId++){
		if(cham_id[newId]==id_layer){
		  itemvector[istrip].xtalk_slope_right=slope_r[istrip];
		  itemvector[istrip].xtalk_intercept_right=intercept_r[istrip];
		  itemvector[istrip].xtalk_chi2_right=chi2_r[istrip];
		  itemvector[istrip].xtalk_slope_left=slope_l[istrip];
		  itemvector[istrip].xtalk_intercept_left=intercept_l[istrip];
		  itemvector[istrip].xtalk_chi2_left=chi2_l[istrip];
		  cncrosstalk->crosstalk[cham_id[newId]]=itemvector;
		}
	      }
	    }
	  }
	}
      }
    }
  }
}


CSCFrontierCrosstalkConditions::CSCFrontierCrosstalkConditions(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  crosstalk.prefillCrosstalkMap();
  // added by Zhen (changed since 1_2_0)
  setWhatProduced(this,&CSCFrontierCrosstalkConditions::produceCrosstalk);
  findingRecord<CSCcrosstalkRcd>();
  //now do what ever other initialization is needed
}


CSCFrontierCrosstalkConditions::~CSCFrontierCrosstalkConditions()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCFrontierCrosstalkConditions::ReturnType
CSCFrontierCrosstalkConditions::produceCrosstalk(const CSCcrosstalkRcd& iRecord)
{
    crosstalk.prefillCrosstalkMap();
    // Added by Zhen, need a new object so to not be deleted at exit
    CSCcrosstalk* mydata=new CSCcrosstalk(crosstalk.get());
    
    return mydata;

}

 void CSCFrontierCrosstalkConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }
