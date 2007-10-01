#ifndef _CALO_MISCALIB_MAP_ECAL
#define _CALO_MISCALIB_MAP_ECAL
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMap.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include <iostream>
#include <iomanip>
#include <map>
#include <vector>



class CaloMiscalibMapEcal: public CaloMiscalibMap {
public:
CaloMiscalibMapEcal(){ 
}

void prefillMap(){
  

   for(int iEta=-EBDetId::MAX_IETA; iEta<=EBDetId::MAX_IETA ;++iEta) {
     if(iEta==0) continue;
     for(int iPhi=EBDetId::MIN_IPHI; iPhi<=EBDetId::MAX_IPHI; ++iPhi) {
       try 
	 {
	   EBDetId ebdetid(iEta,iPhi);
	   map_.setValue(ebdetid.rawId(),1.0);
	 }
       catch (...)
	 {
	 }
     }
   }
   
   for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
     for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
       try 
	 {
 	   EEDetId eedetidpos(iX,iY,1);
	   map_.setValue(eedetidpos.rawId(),1.0);
	   EEDetId eedetidneg(iX,iY,-1);
	   map_.setValue(eedetidneg.rawId(),1.0);
	 }
       catch (...)
	 {
	 }
     }
   }


}


virtual void addCell(const DetId &cell, float scaling_factor)
{
map_.setValue(cell.rawId(),scaling_factor);
}

void print()
 {
 
// std::map<uint32_t,float>::const_iterator it;
// 
//   for(it=map_.getMap().begin();it!=map_.getMap().end();it++){
//   }
 
}

const EcalIntercalibConstants & get(){
return map_;
}

private:

EcalIntercalibConstants map_;
const CaloSubdetectorGeometry *geometry;
};

#endif
