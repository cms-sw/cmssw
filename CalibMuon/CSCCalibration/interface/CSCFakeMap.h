#ifndef _CSC_FAKE_MAP
#define _CSC_FAKE_MAP

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

class CSCFakeMap{
 public:
  CSCFakeMap(){ 
  }
  
  
  void prefillMap(){
    
    const CSCDetId& detId = CSCDetId();
    cn = new CSCGains();

    int max_istrip,id_layer;

    for(int iendcap=detId.minEndcapId(); iendcap<=detId.maxEndcapId(); iendcap++){
      for(int istation=detId.minStationId() ; istation<=detId.maxStationId(); istation++){
	if(istation==1)                detId.maxRingId()==4;
	if(istation==2 || istation==3) detId.maxRingId()==2;
	if(istation==4)                detId.maxRingId()==1;
	
	for(int iring=detId.minRingId(); iring<=detId.maxRingId(); iring++){
	  for(int ichamber=detId.minChamberId(); ichamber<=detId.maxChamberId(); ichamber++){
	    for(int ilayer=detId.minLayerId(); ilayer<=detId.maxLayerId(); ilayer++){
	      if(istation==1 && iring==3){
		max_istrip=64;
	      }else{
		max_istrip=80;
		std::vector<CSCGains::Item> itemvector;
		itemvector.resize(max_istrip);
		
		for(int istrip=0;istrip<max_istrip;istrip++){
		  
		  itemvector[istrip].gain_slope=8.0;
		  itemvector[istrip].gain_intercept=-10.0;
		  itemvector[istrip].gain_chi2=1.0;
		  id_layer = 100000*iendcap+10000*istation+1000*iring+100*ichamber+10*ilayer;
		  cn->gains[id_layer]=itemvector;
		}
	      }
	    }
	  }
	}
      }
    }
    
  }

  
  const CSCGains & get(){
    return (*cn);
  }
  
 private:
  
  CSCGains *cn ; 
  const CSCGeometry *geometry;
  
};

#endif
