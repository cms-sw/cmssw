#ifndef _CSC_FAKE_PEDESTALS_MAP
#define _CSC_FAKE_PEDESTALS_MAP

#include <iostream>
#include <map>
#include <vector>
#include <iomanip>

#include "CondFormats/CSCObjects/interface/CSCPedestals.h"
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>
#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <CondFormats/CSCObjects/interface/CSCReadoutMapping.h>

class CSCFakePedestalsMap{
 public:
  CSCFakePedestalsMap(){ 
  }
  
  float meanped,meanrms;
  int seed;long int M;
  
  void prefillPedestalsMap();
    
  const CSCPedestals & get(){
    return (*cnpedestals);
  }
  
  
 private:
  
  CSCPedestals *cnpedestals ;
  const CSCGeometry *geometry;
  
};

#endif
