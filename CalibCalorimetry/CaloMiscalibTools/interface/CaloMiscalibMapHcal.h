#ifndef _CALO_MISCALIB_MAP_HCAL
#define _CALO_MISCALIB_MAP_HCAL
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMap.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include <iostream>
#include <iomanip>
#include <map>
#include <vector>

class CaloMiscalibMapHcal: public CaloMiscalibMap {
public:
    CaloMiscalibMapHcal(){
    }

void prefillMap(const HcalTopology & topology){

  for (int det = 1; det < 5; det++) {
    for (int eta = -63; eta < 64; eta++) {
      for (int phi = 0; phi < 128; phi++) {
	for (int depth = 1; depth < 5; depth++) {

	  try {
	    HcalDetId hcaldetid ((HcalSubdetector) det, eta, phi, depth);
	    if (topology.valid(hcaldetid))
	    //	    mapHcal_.setValue(hcaldetid.rawId(),1.0);
	    {
	      mapHcal_[hcaldetid.rawId()]=1.0; 
	      //	      std::cout << "Valid cell found: " << det << " " << eta << " " << phi << " " << depth << std::endl;
	    }
		
	  }
	  catch (...) {
	  }
	}
      }
    }
  }
}


virtual void addCell(const DetId &cell, float scaling_factor)
{
  //mapHcal_.setValue(cell.rawId(),scaling_factor);
  mapHcal_[cell.rawId()]=scaling_factor;
}

void print()
 {
 
 std::map<uint32_t,float>::const_iterator it;
 
 //   for(it=mapHcal_.getMap().begin();it!=mapHcal_.getMap().end();it++){
 //   }
   for(it=mapHcal_.begin();it!=mapHcal_.end();it++){
   }
 
}

const std::map<uint32_t, float> & get(){
return mapHcal_;
}

private:

   std::map<uint32_t, float> mapHcal_;
   // EcalIntercalibConstants map_;
   // const CaloSubdetectorGeometry *geometry;
};

#endif
