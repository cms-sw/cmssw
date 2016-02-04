/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/10/26 15:37:08 $
 *  $Revision: 1.5 $
 *  \author M. Maggi -- INFN
 */

#include "RPCCluster.h"
#include "RecoLocalMuon/RPCRecHit/src/RPCRecHitStandardAlgo.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Exception.h"


RPCRecHitStandardAlgo::RPCRecHitStandardAlgo(const edm::ParameterSet& config) :
  RPCRecHitBaseAlgo(config) 
{
}



RPCRecHitStandardAlgo::~RPCRecHitStandardAlgo()
{
}



void RPCRecHitStandardAlgo::setES(const edm::EventSetup& setup) {
}



// First Step
bool RPCRecHitStandardAlgo::compute(const RPCRoll& roll,
				    const RPCCluster& cluster,
				    LocalPoint& Point,
				    LocalError& error)  const
{
  // Get Average Strip position
  float fstrip = (roll.centreOfStrip(cluster.firstStrip())).x();
  float lstrip = (roll.centreOfStrip(cluster.lastStrip())).x();
  float centreOfCluster = (fstrip + lstrip)/2;

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

  // Glob Pos and angle not used so far...
  if (globPos.z()<0){ } // Fake use to avoid warnings
  if (angle<0.){ }      // Fake use to avoid warnings
  this->compute(roll,cl,Point,error);
  return true;
}

