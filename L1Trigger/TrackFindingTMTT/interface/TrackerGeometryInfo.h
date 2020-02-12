#ifndef __TRACKERGEOMETRYINFO_H__
#define __TRACKERGEOMETRYINFO_H__

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"

#include <vector>

using namespace std;

class TrackerTopology;
class TrackerGeometry;

namespace TMTT {


/** 
 * ========================================================================================================
 * Class for storing info on tracker geometry
 * Currently stores the z/r and B of tilted modules, and only needs to do this once per job
 * ========================================================================================================
 */

class TrackerGeometryInfo {

public:
  
  // Initialize.
  TrackerGeometryInfo();

  ~TrackerGeometryInfo() {}

  void getTiltedModuleInfo(const Settings* settings, const TrackerTopology*  trackerTopo, const TrackerGeometry* trackerGeom);
  double barrelNTiltedModules() const { return barrelNTiltedModules_; }
  double barrelNLayersWithTiltedModules() const { return barrelNLayersWithTiltedModules_; }

  vector< double > moduleZoR() const { return moduleZoR_; }
  vector< double > moduleB() const { return moduleB_; }

private:
  unsigned int barrelNTiltedModules_;
  unsigned int barrelNLayersWithTiltedModules_;

  std::vector< double > moduleZoR_;
  vector< double > moduleB_;
};

}

#endif
