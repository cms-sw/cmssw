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
  int chamber_id;
  float slope_right,slope_left,intercept_right,intercept_left, chi2_right,chi2_left;
  std::vector<int> cham_id;
  std::vector<float> slope_r;
  std::vector<float> intercept_r;
  std::vector<float> chi2_r;
  std::vector<float> slope_l;
  std::vector<float> intercept_l;
  std::vector<float> chi2_l;

  void prefillCrosstalkMap();

  const CSCcrosstalk & get(){
    return (*cncrosstalk);
  }
    
 private:
  
  CSCcrosstalk *cncrosstalk ;
  const CSCGeometry *geometry;
  
};

#endif
