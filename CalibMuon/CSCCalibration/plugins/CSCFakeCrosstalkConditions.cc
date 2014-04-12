#include <fstream>

#include "CalibMuon/CSCCalibration/interface/CSCFakeCrosstalkConditions.h"

void CSCFakeCrosstalkConditions::prefillCrosstalk(){
  
  const CSCDetId& detId = CSCDetId();
  cncrosstalk = new CSCcrosstalk();
  
  int max_istrip,id_layer,max_ring,max_cham;
  seed = 10000;	
  srand(seed);
  mean=-0.0009, min=0.035, minchi=1.5, M=1000;
 
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
	//station 1 ring 3 has 64 strips per layer instead of 80(minus & plus side!!!)

	for(int ichamber=detId.minChamberId(); ichamber<=max_cham; ichamber++){

 	  for(int ilayer=detId.minLayerId(); ilayer<=detId.maxLayerId(); ilayer++){
	    //station 1 ring 3 has 64 strips per layer instead of 80 
	    if(istation==1 && iring==3)   max_istrip=64;
	    
	    std::vector<CSCcrosstalk::Item> itemvector;
	    itemvector.resize(max_istrip);
	    id_layer = 100000*iendcap + 10000*istation + 1000*iring + 10*ichamber + ilayer;

	    for(int istrip=0;istrip<max_istrip;istrip++){
	      //create fake values
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
  prefillCrosstalk();
  setWhatProduced(this,&CSCFakeCrosstalkConditions::produceCrosstalk);
  findingRecord<CSCcrosstalkRcd>();
  //now do what ever other initialization is needed
}


CSCFakeCrosstalkConditions::~CSCFakeCrosstalkConditions()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete cncrosstalk;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CSCFakeCrosstalkConditions::ReturnType
CSCFakeCrosstalkConditions::produceCrosstalk(const CSCcrosstalkRcd& iRecord)
{
  return cncrosstalk;  
}

 void CSCFakeCrosstalkConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
 edm::ValidityInterval & oValidity)
 {
 oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 
 }


