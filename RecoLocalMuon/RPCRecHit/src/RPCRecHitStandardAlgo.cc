/*
 *  See header file for a description of this class.
 *
 *  \author M. Maggi -- INFN
 */

#include "RPCCluster.h"
#include "RecoLocalMuon/RPCRecHit/src/RPCRecHitStandardAlgo.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"

// First Step
bool RPCRecHitStandardAlgo::compute(const RPCRoll& roll,
                                    const RPCCluster& cluster,
                                    LocalPoint& Point,
                                    LocalError& error,
                                    float& time,
                                    float& timeErr) const {
  // Get Average Strip position
  const float fstrip = (roll.centreOfStrip(cluster.firstStrip())).x();
  const float lstrip = (roll.centreOfStrip(cluster.lastStrip())).x();
  const float centreOfCluster = (fstrip + lstrip) / 2;
  const double y = cluster.hasY() ? cluster.y() : 0;
  Point = LocalPoint(centreOfCluster, y, 0);

  if (!cluster.hasY()) {
    error = LocalError(roll.localError((cluster.firstStrip() + cluster.lastStrip()) / 2.));
  } else {
    // Use the default one for local x error
    float ex2 = roll.localError((cluster.firstStrip() + cluster.lastStrip()) / 2.).xx();
    // Maximum estimate of local y error, (distance to the boundary)/sqrt(3)
    // which gives consistent error to the default one at y=0
    const float stripLen = roll.specificTopology().stripLength();
    const float maxDy = stripLen / 2 - std::abs(cluster.y());

    // Apply x-position correction for the endcap
    if (roll.id().region() != 0) {
      const auto& topo = dynamic_cast<const TrapezoidalStripTopology&>(roll.topology());
      const double angle = topo.stripAngle((cluster.firstStrip() + cluster.lastStrip()) / 2.);
      const double x = centreOfCluster - y * std::tan(angle);
      Point = LocalPoint(x, y, 0);

      // rescale x-error by the change of local pitch
      const double scale = topo.localPitch(Point) / topo.pitch();
      ex2 *= scale * scale;
    }

    error = LocalError(ex2, 0, maxDy * maxDy / 3.);
  }

  if (cluster.hasTime()) {
    time = cluster.time();
    timeErr = cluster.timeRMS();
  } else {
    time = 0;
    timeErr = -1;
  }

  return true;
}

bool RPCRecHitStandardAlgo::compute(const RPCRoll& roll,
                                    const RPCCluster& cl,
                                    const float& angle,
                                    const GlobalPoint& globPos,
                                    LocalPoint& Point,
                                    LocalError& error,
                                    float& time,
                                    float& timeErr) const {
  this->compute(roll, cl, Point, error, time, timeErr);
  return true;
}
