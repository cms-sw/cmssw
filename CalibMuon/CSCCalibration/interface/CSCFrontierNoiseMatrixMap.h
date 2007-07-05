#ifndef _CSC_FRONTIER_NOISEMATRIX_MAP
#define _CSC_FRONTIER_NOISEMATRIX_MAP

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

class CSCFrontierNoiseMatrixMap{
 public:
  CSCFrontierNoiseMatrixMap(){ 
  }

  int old_chamber_id,old_strip,new_chamber_id,new_strip;
  float old_elm33,old_elm34, old_elm44, old_elm35, old_elm45, old_elm55, old_elm46, old_elm56, old_elm66, old_elm57, old_elm67, old_elm77;
  std::vector<int> old_cham_id;
  std::vector<int> old_strips;
  std::vector<float> old_elem33;
  std::vector<float> old_elem34;
  std::vector<float> old_elem44;
  std::vector<float> old_elem45;
  std::vector<float> old_elem35;
  std::vector<float> old_elem55;
  std::vector<float> old_elem46;
  std::vector<float> old_elem56;
  std::vector<float> old_elem66;
  std::vector<float> old_elem57;
  std::vector<float> old_elem67;
  std::vector<float> old_elem77;


  float new_elm33,new_elm34, new_elm44, new_elm35, new_elm45, new_elm55, new_elm46, new_elm56, new_elm66, new_elm57, new_elm67, new_elm77;
  std::vector<int> new_cham_id;
  std::vector<int> new_strips;
  std::vector<float> new_elem33;
  std::vector<float> new_elem34;
  std::vector<float> new_elem44;
  std::vector<float> new_elem45;
  std::vector<float> new_elem35;
  std::vector<float> new_elem55;
  std::vector<float> new_elem46;
  std::vector<float> new_elem56;
  std::vector<float> new_elem66;
  std::vector<float> new_elem57;
  std::vector<float> new_elem67;
  std::vector<float> new_elem77;

  void prefillNoiseMatrixMap();
  
  const CSCNoiseMatrix & get(){
    return (*cnmatrix);
  }
  
  
 private:
  
  CSCNoiseMatrix *cnmatrix ;
  const CSCGeometry *geometry;
  
};

#endif
