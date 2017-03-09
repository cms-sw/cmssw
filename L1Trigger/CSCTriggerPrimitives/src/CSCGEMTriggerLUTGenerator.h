#ifndef L1Trigger_CSCTriggerPrimitives_CSCGEMTriggerLUTGenerator_h
#define L1Trigger_CSCTriggerPrimitives_CSCGEMTriggerLUTGenerator_h

#include "L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h"
#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include <vector>
#include <map>

class CSCGEMTriggerLUTGenerator
{
public:

  CSCGEMTriggerLUTGenerator() {}
  ~CSCGEMTriggerLUTGenerator() {}
  
  /// set CSC and GEM geometries for the matching needs
  void setCSCGeometry(const CSCGeometry *g) { csc_g = g; }
  void setGEMGeometry(const GEMGeometry *g) { gem_g = g; }
  
  void generateLUTsME11(unsigned theEndcap, unsigned theStation, unsigned theSector, unsigned theSubSector, unsigned theTrigChamber);
  void generateLUTsME21(unsigned theEndcap, unsigned theStation, unsigned theSector, unsigned theSubSector, unsigned theTrigChamber);

  int assignGEMRoll(const std::map<int,std::pair<double,double> >&, double eta);
  
 private:
  const CSCGeometry* csc_g;
  const GEMGeometry* gem_g;
};

#endif
