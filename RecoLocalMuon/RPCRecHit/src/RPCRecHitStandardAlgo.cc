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

// First Step
bool RPCRecHitStandardAlgo::compute(const RPCRoll& roll,
				    const RPCCluster& cluster,
				    LocalPoint& Point,
				    LocalError& error,
            float& time, float& timeErr)  const
{
  // Get Average Strip position
  const float fstrip = (roll.centreOfStrip(cluster.firstStrip())).x();
  const float lstrip = (roll.centreOfStrip(cluster.lastStrip())).x();
  const float centreOfCluster = (fstrip + lstrip)/2;

  Point = LocalPoint(centreOfCluster,cluster.y(),0);
  if ( !cluster.hasY() ) {
    error = LocalError(roll.localError((cluster.firstStrip()+cluster.lastStrip())/2.));
  }
  else {
    // Use the default one for local x error
    const float ex2 = roll.localError((cluster.firstStrip()+cluster.lastStrip())/2.).xx();
    // Maximum estimate of local y error, (distance to the boundary)/sqrt(3)
    // which gives consistent error to the default one at y=0
    const float stripLen = roll.specificTopology().stripLength();
    const float maxDy = stripLen/2 - std::abs(cluster.y());
    error = LocalError(ex2, 0, maxDy*maxDy/3.);
  }

  if ( cluster.hasTime() ) {
    time = cluster.time();
    timeErr = cluster.timeRMS();
  }

  return true;
}

bool RPCRecHitStandardAlgo::compute(const RPCRoll& roll,
                                    const RPCCluster& cl,
                                    const float& angle,
                                    const GlobalPoint& globPos,
                                    LocalPoint& Point,
                                    LocalError& error,
                                    float& time, float& timeErr)  const
{
  this->compute(roll,cl,Point,error,time,timeErr);
  return true;
}

