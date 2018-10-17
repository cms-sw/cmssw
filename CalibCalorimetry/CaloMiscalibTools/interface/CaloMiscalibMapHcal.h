#ifndef _CALO_MISCALIB_MAP_HCAL
#define _CALO_MISCALIB_MAP_HCAL
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMap.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
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

  for (int det = 1; det <= HcalForward; det++) {
    for (int eta = -HcalDetId::kHcalEtaMask2; eta <= (int)(HcalDetId::kHcalEtaMask2); eta++) {
      for (unsigned int phi = 1; phi <= HcalDetId::kHcalPhiMask2; phi++) {
	for (unsigned int depth = 1; depth <= HcalDetId::kHcalDepthMask2; depth++) {

	  try {
	    HcalDetId hcaldetid ((HcalSubdetector) det, eta, phi, depth);
	    if (topology.valid(hcaldetid)) {
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


void addCell(const DetId &cell, float scaling_factor) override
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
