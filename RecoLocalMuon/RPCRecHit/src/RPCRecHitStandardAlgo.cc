/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/04/15 21:20:10 $
 *  $Revision: 1.1 $
 *  \author M. Maggi -- INFN
 */

#include "RecoLocalMuon/RPCRecHit/src/RPCRecHitStandardAlgo.h"
#include "DataFormats/MuonDetId/interface/RPCWireId.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCRollService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Exception.h"

RPCRecHitStandardAlgo::RPCRecHitStandardAlgo(const ParameterSet& config) :
  RPCRecHitBaseAlgo(config) 
{
}



RPCRecHitStandardAlgo::~RPCRecHitStandardAlgo()
{
}



void RPCRecHitStandardAlgo::setES(const EventSetup& setup) {
}



// First Step
bool RPCRecHitStandardAlgo::compute(const RPCRoll* roll,
				    const RPCCluster& cluster,
				    LocalPoint& Point,
				    LocalError& error) const 
{
  // Get the detId
  RPCDetId detId = roll->id();
  RPCService rpcServ(roll);

  // Get Average Strip position
  LocalPoint Posint(roll->Topology().wirePosition(cluster.wire()), 0, 0);
  const GlobalPoint globWirePos = roll->surface().toGlobal(locWirePos);
  
  return compute(roll, wireId, cluster.time(), globWirePos, leftPoint, rightPoint, error, 1); 
}



// Second step: the same as 1st step
bool RPCRecHitStandardAlgo::compute(const RPCRoll* roll,
				const RPCRecHit1D& recHit1D,
				const float& angle,
				RPCRecHit1D& newHit1D) const {
  newHit1D.setPositionAndError(recHit1D.localPosition(), recHit1D.localPositionError());
  return true;
}



// Third step.
bool RPCRecHitStandardAlgo::compute(const RPCRoll* roll,
				const RPCRecHit1D& recHit1D,
				const float& angle,
				const GlobalPoint& globPos, 
				RPCRecHit1D& newHit1D) const {
  return compute(roll, recHit1D.wireId(), recHit1D.clusterTime(), globPos, newHit1D, 3);
}



// Do the actual work.
bool RPCRecHitStandardAlgo::compute(const RPCRoll* roll,
				const RPCWireId& wireId,
				const float clusterTime,
				const GlobalPoint& globPos, 
				LocalPoint& leftPoint,
				LocalPoint& rightPoint,
				LocalError& error,
				int step) const {
  // Subtract the offset to the cluster time accordingly to the RPCTTrigBaseSync concrete instance
  float driftTime = clusterTime - theSync->offset(roll, wireId, globPos); 
  
  // check for out-of-time
  if (driftTime < minTime || driftTime > maxTime) {
    if (debug) cout << "[RPCRecHitStandardAlgo]*** Drift time out of window for in-time hits "
			      << driftTime << endl;
    // Hits are interpreted as coming from out-of-time pile-up and recHit
    // is ignored.
    return false;
  }

  // Small negative times interpreted as hits close to the wire.
  if (driftTime<0.) driftTime=0;

  // Compute the drift distance
  float drift = driftTime * vDrift;

  // Get Wire position
  LocalPoint locWirePos(roll->specificTopology().wirePosition(wireId.wire()), 0, 0);
  //Build the two possible points and the error on the position
  leftPoint  = LocalPoint(locWirePos.x()-drift,
                            locWirePos.y(),
                            locWirePos.z());
  rightPoint = LocalPoint(locWirePos.x()+drift,
                            locWirePos.y(),
                            locWirePos.z());
  error = LocalError(hitResolution*hitResolution,0.,0.);


  if(debug) {
    cout << "[RPCRecHitStandardAlgo] Compute drift distance, for cluster at wire: " << wireId << endl
	 << "       Step:           " << step << endl
	 << "       Cluster time:      " << clusterTime << endl
	 << "       Drift time:     " << driftTime << endl
	 << "       Drift distance: " << drift << endl
	 << "       Hit Resolution: " << hitResolution << endl
	 << "       Left point:     " << leftPoint << endl
	 << "       Right point:    " << rightPoint << endl
	 << "       Error:          " << error << endl;
   }
  
  return true;
  
}


// Interface to the method which does the actual work suited for 2nd and 3rd steps 
bool RPCRecHitStandardAlgo::compute(const RPCRoll* roll,
				const RPCWireId& wireId,
				const float clusterTime,
				const GlobalPoint& globPos, 
				RPCRecHit1D& newHit1D,
				int step) const {
  LocalPoint leftPoint;
  LocalPoint rightPoint;
  LocalError error;

  if(compute(roll, wireId, clusterTime, globPos, leftPoint, rightPoint, error, step)) {
    // Set the position and the error of the rechit which is being updated
    switch(newHit1D.lrSide()) {
	
    case RPCEnums::Left:
      newHit1D.setPositionAndError(leftPoint, error);
      break;
	
    case RPCEnums::Right:
      newHit1D.setPositionAndError(rightPoint, error);
      break;
	
    default:
      throw cms::Exception("InvalidRPCCellSide") << "[RPCRecHitStandardAlgo] Compute at Step "
						<< step << ", Hit side "
						<< newHit1D.lrSide()
						<< " is invalid!" << endl;
      return false;
    }
      
    return true;
  }else {
    return false;
  }
}


float RPCRecHitStandardAlgo::vDrift;

  
float RPCRecHitStandardAlgo::hitResolution;

  
float RPCRecHitStandardAlgo::minTime;

  
float RPCRecHitStandardAlgo::maxTime;

  
bool RPCRecHitStandardAlgo::debug;
