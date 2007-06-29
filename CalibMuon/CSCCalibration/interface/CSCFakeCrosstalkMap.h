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
  
  float mean,min,minchi;
  int seed;long int M;
  
  void prefillCrosstalkMap();

  const CSCcrosstalk & get(){
    return (*cncrosstalk);
  }
    
 private:
  
  CSCcrosstalk *cncrosstalk ;
  const CSCGeometry *geometry;
  
};

#endif
