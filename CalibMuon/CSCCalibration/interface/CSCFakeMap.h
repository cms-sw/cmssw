#ifndef _CSC_FAKE_MAP
#define _CSC_FAKE_MAP

#include <iostream>
#include <map>
#include <vector>
#include <iomanip>

#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>
#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <CondFormats/CSCObjects/interface/CSCReadoutMapping.h>
#include <CondFormats/CSCObjects/interface/CSCGains.h>
#include "CondFormats/CSCObjects/interface/CSCcrosstalk.h"
#include "CondFormats/CSCObjects/interface/CSCPedestals.h"
#include "CondFormats/CSCObjects/interface/CSCNoiseMatrix.h"
#include <CalibMuon/CSCCalibration/interface/FakeMap.h>

class CSCFakeMap:public FakeMap{

 public:
  CSCFakeMap(){
  }

  void prefillMap(){
    
    const CSCDetId& detId = CSCDetId();

    for(int iendcap=detId.minEndcapId(); iendcap<=detId.maxEndcapId(); iendcap++){
      for(int istation=detId.minStationId() ; istation<=detId.maxStationId(); istation++){
	for(int iring=detId.minRingId(); iring<=detId.maxRingId(); iring++){
	  for(int ichamber=detId.minChamberId(); ichamber<=detId.maxChamberId(); ichamber++){
	    for(int ilayer=detId.minLayerId(); ilayer<=detId.maxLayerId(); ilayer++){
	      try{
		CSCDetId cscdetid(iendcap,istation,iring,ichamber,ilayer);
		map_.setValue(cscdetid.rawId(),1.0);map_.setValue(cscdetid.rawId(),1.0);
	      }
	      catch(...)
		{
		}
	    }
	  }
	}
      }
    }


  }
  
  virtual void csc(const DetId &cell, float scaling_factor)
    {
      map_.setValue(cell.rawId(),scaling_factor);
    }
  
  void print()
    {
      
      std::map<uint32_t,float>::const_iterator it;
      
      for(it=map_.getMap().begin();it!=map_.getMap().end();it++){
      }
      
    }
  
  const CSCGains & get(){
    return map_;
  }
  
 private:
  CSCGains map_ ;
  const CSCGeometry *geometry;
  
};


#endif
