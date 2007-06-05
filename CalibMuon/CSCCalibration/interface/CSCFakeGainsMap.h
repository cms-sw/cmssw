#ifndef _CSC_FAKE_GAINS_MAP
#define _CSC_FAKE_GAINS_MAP

#include <iostream>
#include <map>
#include <vector>
#include <iomanip>

#include "CondFormats/CSCObjects/interface/CSCGains.h"
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>
#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <CondFormats/CSCObjects/interface/CSCReadoutMapping.h>

class CSCFakeGainsMap{
 public:
  CSCFakeGainsMap(){ 
  }
  

  void prefillGainsMap(){
    
    const CSCDetId& detId = CSCDetId();
    cngains = new CSCGains();
    
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
	      
	      std::vector<CSCGains::Item> itemvector;
	      itemvector.resize(max_istrip);
	      id_layer = 100000*iendcap + 10000*istation + 1000*iring + 10*ichamber + ilayer;

	      for(int istrip=0;istrip<max_istrip;istrip++){
		itemvector[istrip].gain_slope    = 7.55;
		itemvector[istrip].gain_intercept= -10.00;
		itemvector[istrip].gain_chi2     = 2.00;
		cngains->gains[id_layer]=itemvector;
	      }
	    }
	  }
	}
      }
    }
  }

  const CSCGains & get(){
    return (*cngains);
  }
  
  
 private:
  CSCGains *cngains ;
  const CSCGeometry *geometry; 
};

#endif
