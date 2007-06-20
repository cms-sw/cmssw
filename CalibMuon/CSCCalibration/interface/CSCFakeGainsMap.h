#ifndef _CSC_FAKE_GAINS_MAP
#define _CSC_FAKE_GAINS_MAP

#include <iostream>
#include <map>
#include <vector>
#include <iomanip>
#include <sys/stat.h>

#include "CondFormats/CSCObjects/interface/CSCGains.h"
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>
#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <CondFormats/CSCObjects/interface/CSCReadoutMapping.h>

class CSCFakeGainsMap{
 public:

  CSCFakeGainsMap(){ 
  }
  
  float mean,min,minchi;
  int seed;long int M;

  const CSCGains & get(){
    return (*cngains);
  }
  
  void prefillGainsMap();  
  
 private:
  const CSCGeometry *geometry; 
  CSCGains *cngains ;
  
};

#endif
