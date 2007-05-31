#ifndef _CSC_FAKE_MAP_H
#define _CSC_FAKE_MAP_H

#include <iostream>
#include <map>
#include <vector>
#include <iomanip>

#include "DataFormats/DetId/interface/DetId.h"
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>
#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <CondFormats/CSCObjects/interface/CSCReadoutMapping.h>



class CSCFakeMap{

public:
CSCFakeMap(){
}

 void prefillMap(){

   const CSCDetId& detId = CSCDetId();

   /*
   for(detId.station()=1; detId.station()<=4;detId.station()++){
     for(detId.ring()=1; detId.ring()<=4;detId.ring()++){
       try{
	 detId cscdetid(detId.station(),detId.ring());
	 map_.setValue(cscdetid.rawId(),1.0);
       }
       catch(...)
	 {
	 }
     }
   }
   */

   if(detId.station() < 4 && (detId.ring() <=4) )
     {
       // detId cscdetid(detId.station(),detId.ring());
       map_.setValue(detId.station().rawId(),1.0);
       map_.setValue(detId.ring().rawId(),1.0);
     }

 }
 
 void print()
   {
     
     std::map<uint32_t,float>::const_iterator it;
     
     for(it=map_.getMap().begin();it!=map_.getMap().end();it++){
     }
     
   }
 
 const CSCobject & get(){
   return map_;
 }
 
 public:
 
 
};


#endif
