#ifndef _CSC_FAKE_NOISEMATRIX_MAP
#define _CSC_FAKE_NOISEMATRIX_MAP

#include <iostream>
#include <map>
#include <vector>
#include <iomanip>

#include "CondFormats/CSCObjects/interface/CSCNoiseMatrix.h"
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>
#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <CondFormats/CSCObjects/interface/CSCReadoutMapping.h>

class CSCFakeNoiseMatrixMap{
 public:
  CSCFakeNoiseMatrixMap(){ 
  }
  
  void prefillNoiseMatrixMap();
  
  const CSCNoiseMatrix & get(){
    return (*cnmatrix);
  }
  
  
 private:
  
  CSCNoiseMatrix *cnmatrix ;
  const CSCGeometry *geometry;
  
};

#endif
