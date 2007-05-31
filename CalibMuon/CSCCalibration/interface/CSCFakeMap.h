#ifndef _CSCFAKEMAP_H
#define _CSCFAKEMAP_H

#include <iostream>
#include "DataFormats/DetId/interface/DetId.h"
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>
#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <CondFormats/CSCObjects/interface/CSCReadoutMapping.h>

#include <map>
#include <vector>
#include <iostream>

class CSCFakeMap{

public:
CSCFakeMap(){}

 void prefillMap(){

   if(detId.station() < 4 && (detId.ring() <=4) )
     {
       map_.setValue(gains.rawId(),1.0);
       map_.setValue(crosstalk.rawId(),1.0);
       map_.setValue(matrix.rawId(),1.0);
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
