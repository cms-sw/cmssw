/*
 *  See header file for a description of this class.
 *
 *  \author M. Maggi -- INFN
 */

#include "RPCCluster.h"
#include "RecoLocalMuon/RPCRecHit/src/RPCRecHitStandardAlgo.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Exception.h"

// First Step
bool RPCRecHitStandardAlgo::compute(const RPCRoll& roll,
				    const RPCCluster& cluster,
				    LocalPoint& Point,
				    LocalError& error)  const
{
  // Get Average Strip position
  const float fstrip = (roll.centreOfStrip(cluster.firstStrip())).x();
  const float lstrip = (roll.centreOfStrip(cluster.lastStrip())).x();
  const float centreOfCluster = (fstrip + lstrip)/2;

  LocalPoint loctemp2(centreOfCluster,0.,0.);

  Point = loctemp2;
  error = roll.localError((cluster.firstStrip()+cluster.lastStrip())/2.);

  return true;
}

bool RPCRecHitStandardAlgo::compute(const RPCRoll& roll,
                                    const RPCCluster& cl,
                                    const float& angle,
                                    const GlobalPoint& globPos,
                                    LocalPoint& Point,
                                    LocalError& error)  const
{
  this->compute(roll,cl,Point,error);
  return true;
}

