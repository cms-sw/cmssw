/*
 *
* This is a part of CTPPS offline software.
* Author:
*   Fabrizio Ferro (ferro@ge.infn.it)
*   Enrico Robutti (robutti@ge.infn.it)
*   Fabio Ravera   (fabio.ravera@cern.ch)
*
*/
#ifndef RecoPPS_Local_RPixRoadFinder_H
#define RecoPPS_Local_RPixRoadFinder_H

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CTPPSReco/interface/CTPPSPixelCluster.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelRecHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "RecoPPS/Local/interface/RPixClusterToHit.h"
#include "RecoPPS/Local/interface/RPixDetPatternFinder.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "Geometry/VeryForwardRPTopology/interface/RPTopology.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"
#include "Geometry/Records/interface/VeryForwardMisalignedGeometryRecord.h"

#include <vector>
#include <set>

class RPixRoadFinder : public RPixDetPatternFinder {
public:
  explicit RPixRoadFinder(const edm::ParameterSet &param);
  ~RPixRoadFinder() override;
  void findPattern(bool *is2planepot) override;

private:
  int verbosity_;
  double roadRadius_;
  unsigned int minRoadSize_;
  unsigned int maxRoadSize_;
  double roadRadiusBadPot_;
  void run(const edm::DetSetVector<CTPPSPixelRecHit> &input, const CTPPSGeometry &geometry, std::vector<Road> &roads);
};

#endif
