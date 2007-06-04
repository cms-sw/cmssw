#ifndef _CSC_FAKE_CROSSTALK_MAP
#define _CSC_FAKE_CROSSTALK_MAP

#include <iostream>
#include <map>
#include <vector>
#include <iomanip>

#include "CondFormats/CSCObjects/interface/CSCcrosstalk.h"
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>
#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <CondFormats/CSCObjects/interface/CSCReadoutMapping.h>

class CSCFakeCrosstalkMap{
 public:
  CSCFakeCrosstalkMap(){ 
  }
  
  
  void prefillCrosstalkMap(){
    
    const CSCDetId& detId = CSCDetId();
    cncrosstalk = new CSCcrosstalk();

    int max_istrip,id_layer;
    //endcap=1 to 2,station=1 to 4, ring=1 to 4,chamber=1 to 36,layer=1 to 6 
    for(int iendcap=detId.minEndcapId(); iendcap<=detId.maxEndcapId(); iendcap++){
      for(int istation=detId.minStationId() ; istation<=detId.maxStationId(); istation++){
	if(istation==1)                detId.maxRingId()==4;
	if(istation==2 || istation==3) detId.maxRingId()==2;
	if(istation==4)                detId.maxRingId()==1;
	
	for(int iring=detId.minRingId(); iring<=detId.maxRingId(); iring++){
	  // std::cout <<"Station: "<<iendcap<<" and ring "<<iring<<std::endl;
	  for(int ichamber=detId.minChamberId(); ichamber<=detId.maxChamberId(); ichamber++){
	    for(int ilayer=detId.minLayerId(); ilayer<=detId.maxLayerId(); ilayer++){
	      if(istation==1 && iring==3){
		max_istrip=64;
	      }else{
		max_istrip=80;
		std::vector<CSCcrosstalk::Item> itemvector;
		itemvector.resize(max_istrip);
		
		for(int istrip=0;istrip<max_istrip;istrip++){
		  
		  itemvector[istrip].xtalk_slope_right=8.0;
		  itemvector[istrip].xtalk_intercept_right=8.0;
		  itemvector[istrip].xtalk_chi2_right=8.0;
		  itemvector[istrip].xtalk_slope_left=8.0;
		  itemvector[istrip].xtalk_intercept_left=-10.0;
		  itemvector[istrip].xtalk_chi2_left=1.0;

		  id_layer = 100000*iendcap+10000*istation+1000*iring+100*ichamber+10*ilayer+ilayer;
		  std::cout<<" ID is: "<<id_layer<<std::endl;
		  cncrosstalk->crosstalk[id_layer]=itemvector;
		}
	      }
	    }
	  }
	}
      }
    }
    
  }

  const CSCcrosstalk & get(){
    return (*cncrosstalk);
  }
  
  
 private:
  
  CSCcrosstalk *cncrosstalk ;
  const CSCGeometry *geometry;
  
};

#endif
