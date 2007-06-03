#ifndef _CSC_FAKE_MAP
#define _CSC_FAKE_MAP
#include <iostream>
#include <map>
#include <vector>
#include <iomanip>

#include "CondFormats/CSCObjects/interface/CSCGains.h"
#include "CalibMuon/CSCCalibration/interface/FakeMap.h"
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
    CSCobject *cn = new CSCobject();
    int max_istrip;
    
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
		for(int istrip=0;istrip<max_istrip;istrip++){
		  CSCDetId cscdetid(iendcap,istation,iring,ichamber,ilayer);
		  
		  cn->obj[ilayer][istrip].resize(2);
		  cn->obj[ilayer][istrip][0] = 8.0;
		  cn->obj[ilayer][istrip][1] = -10.0;
		  gains.setValue(cscdetid.rawId(),1.0);
		}
	      }
	    }
	  }
	}
      }
    }
    
  }

  

/*
virtual void csc(const DetId &cell, float scaling_factor)
{
  CSCGains.setValue(cell.rawId(),scaling_factor);
}
*/

    /*
    void print()
      {
	//	std::map<int,std::vector< > > gains;  
	std::map<uint32_t,float>::const_iterator it;
	
	for(it=CSCGains.begin();it!=CSCGains.end();it++){
	}
	
      }
    
    */

    const CSCGains & get(){
      return gains;
    }
    
 private:
    
    //std::map<uint32_t, float> gains;
    //CSCGains gains;
    const CSCGeometry *geometry;
    
};

#endif
