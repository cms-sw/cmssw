/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/02/15 13:54:45 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */

#include "RecoLocalMuon/DTRecHit/src/DTLinearDriftAlgo.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalMuon/DTRecHit/interface/DTTTrigBaseSync.h"

using namespace std;

DTLinearDriftAlgo::DTLinearDriftAlgo(const edm::ParameterSet& config) :
  DTRecHitBaseAlgo(config) {
    // Get the Drift Velocity from parameter set. 
    vDrift = config.getParameter<double>("driftVelocity"); // FIXME: Default was 0.00543 cm/ns

    minTime = config.getParameter<double>("minTime"); // FIXME: Default was -3 ns

    maxTime = config.getParameter<double>("maxTime"); // FIXME: Default was 415 ns

    hitResolution = config.getParameter<double>("hitResolution"); // FIXME: Default is 
    // Set verbose output
    debug = config.getUntrackedParameter<bool>("debug");
    
  }



DTLinearDriftAlgo::~DTLinearDriftAlgo(){}



// First Step
bool DTLinearDriftAlgo::compute(const DTLayer* layer,
				const DTDigi& digi,
				LocalPoint& leftPoint,
				LocalPoint& rightPoint,
				LocalError& error) const {
  // Get the layerId
  DTLayerId layerId = layer->id();//FIXME: pass it instead of get it from layer
  const DTWireId wireId(layerId, digi.wire());

  // Get Wire position
  LocalPoint locWirePos(layer->specificTopology().wirePosition(wireId.wire()), 0, 0);
  const GlobalPoint globWirePos = layer->surface().toGlobal(locWirePos);


  // Note that for TOF and delays for signal propagation along the wire
  // the digis is assumed to be at the wire center
  float driftTime = digi.time() - theSync->offset(layer, wireId, globWirePos); 
  
  // check for out-of-time
  if (driftTime < minTime || driftTime > maxTime) {
    if (debug) cout << "[DTLinearDriftAlgo]*** Drift time out of window for in-time hits "
			      << driftTime << endl;
    // Hits are interpreted as coming from out-of-time pile-up and recHit
    // is ignored.
    return false;
  }

  // Small negative times interpreted as hits close to the wire.
  if (driftTime<0.) driftTime=0;

  // Compute the drift distance
  float drift = driftTime * vDrift;


  //Build the two possible points and the error on the position
  leftPoint  = LocalPoint(locWirePos.x()-drift,
                            locWirePos.y(),
                            locWirePos.z());
  rightPoint = LocalPoint(locWirePos.x()+drift,
                            locWirePos.y(),
                            locWirePos.z());
  error = LocalError(hitResolution*hitResolution,0.,0.);


  if(debug) {
    cout << "[DTLinearDriftAlgo] Compute drift distance, for digi at wire: " << wireId << endl
	 << "       Digi time:      " << digi.time() << endl
	 << "       Drift time:     " << driftTime << endl
	 << "       Drift distance: " << drift << endl
	 << "       Hit Resolution: " << hitResolution << endl
	 << "       Left point:     " << leftPoint << endl
	 << "       Right point:    " << rightPoint << endl
	 << "       Error:          " << error << endl;
   }
  
  return true;
}


// Second step: the same as 1st step
bool DTLinearDriftAlgo::compute(const DTLayer* layer,
				const DTDigi& digi,
				const float& angle,
				LocalPoint& leftPoint,
				LocalPoint& rightPoint,
				LocalError& error) const {
  // FIXME: What to do?
  return compute(layer, digi, leftPoint, rightPoint, error);
}



// Third step: the same as 1st step
bool DTLinearDriftAlgo::compute(const DTLayer* layer,
				const DTDigi& digi,
				const float& angle,
				const GlobalPoint& globPos, 
				LocalPoint& leftPoint,
				LocalPoint& rightPoint,
				LocalError& error) const {
  // FIXME: What to do?
  return compute(layer, digi, leftPoint, rightPoint, error);
}


float DTLinearDriftAlgo::vDrift;

  
float DTLinearDriftAlgo::hitResolution;

  
float DTLinearDriftAlgo::minTime;

  
float DTLinearDriftAlgo::maxTime;

  
bool DTLinearDriftAlgo::debug;
