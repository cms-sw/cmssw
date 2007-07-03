#ifndef _CSC_FRONTIER_CROSSTALK_MAP
#define _CSC_FRONTIER_CROSSTALK_MAP

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

class CSCFrontierCrosstalkMap{
 public:
  CSCFrontierCrosstalkMap(){ 
  }
  
  float mean,min,minchi;
  int seed;long int M;
  int old_chamber_id,old_strip,new_chamber_id,new_strip;
  float old_slope_right,old_slope_left,old_intercept_right,old_intercept_left, old_chi2_right,old_chi2_left;
  std::vector<int> old_cham_id;
  std::vector<int> old_strips;
  std::vector<float> old_slope_r;
  std::vector<float> old_intercept_r;
  std::vector<float> old_chi2_r;
  std::vector<float> old_slope_l;
  std::vector<float> old_intercept_l;
  std::vector<float> old_chi2_l;
  float new_slope_right,new_slope_left,new_intercept_right,new_intercept_left, new_chi2_right,new_chi2_left;
  std::vector<int> new_cham_id;
  std::vector<int> new_strips;
  std::vector<float> new_slope_r;
  std::vector<float> new_intercept_r;
  std::vector<float> new_chi2_r;
  std::vector<float> new_slope_l;
  std::vector<float> new_intercept_l;
  std::vector<float> new_chi2_l;

  void prefillCrosstalkMap();

  const CSCcrosstalk & get(){
    return (*cncrosstalk);
  }
    
 private:
  
  CSCcrosstalk *cncrosstalk ;
  const CSCGeometry *geometry;
  
};

#endif
