#ifndef _CSC_FRONTIER_GAINS_MAP
#define _CSC_FRONTIER_GAINS_MAP

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

class CSCFrontierGainsMap{
 public:

  CSCFrontierGainsMap(){ 
  }
  
  float mean,min,minchi;
  int seed;long int M;
  int old_chamber_id,old_strip,new_chamber_id,new_strip;
  float old_gainslope,old_intercpt, old_chisq;
  std::vector<int> old_cham_id;
  std::vector<int> old_strips;
  std::vector<float> old_slope;
  std::vector<float> old_intercept;
  std::vector<float> old_chi2;
  float new_gainslope,new_intercpt, new_chisq;
  std::vector<int> new_cham_id;
  std::vector<int> new_strips;
  std::vector<float> new_slope;
  std::vector<float> new_intercept;
  std::vector<float> new_chi2;
  const CSCGains & get(){
    return (*cngains);
  }
  
  void prefillGainsMap();  
  
 private:
  const CSCGeometry *geometry; 
  CSCGains *cngains ;
  
};

#endif
