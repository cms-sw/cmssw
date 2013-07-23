/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/04/30 09:30:06 $
 *  $Revision: 1.3 $
 *  \author G. Cerminara - INFN Torino
 */

#include "RecoLocalMuon/DTRecHit/plugins/DTLinearDriftAlgo.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;
using namespace edm;

DTLinearDriftAlgo::DTLinearDriftAlgo(const ParameterSet& config) :
  DTRecHitBaseAlgo(config) {
    // Get the Drift Velocity from parameter set. 
    vDrift = config.getParameter<double>("driftVelocity"); // FIXME: Default was 0.00543 cm/ns
    // vDriftMB1W1 = config.getParameter<double>("driftVelocityMB1W1"); // FIXME: Default was 0.00543 cm/ns

    minTime = config.getParameter<double>("minTime"); // FIXME: Default was -3 ns

    maxTime = config.getParameter<double>("maxTime"); // FIXME: Default was 415 ns

    hitResolution = config.getParameter<double>("hitResolution"); // FIXME: Default is 
    // Set verbose output
    debug = config.getUntrackedParameter<bool>("debug");
    
  }



DTLinearDriftAlgo::~DTLinearDriftAlgo(){}



void DTLinearDriftAlgo::setES(const EventSetup& setup) {
  theSync->setES(setup);
}



// First Step
bool DTLinearDriftAlgo::compute(const DTLayer* layer,
				const DTDigi& digi,
				LocalPoint& leftPoint,
				LocalPoint& rightPoint,
				LocalError& error) const {
  // Get the wireId
  DTLayerId layerId = layer->id();
  const DTWireId wireId(layerId, digi.wire());

  // Get Wire position
  if(!layer->specificTopology().isWireValid(digi.wire())) return false;
  LocalPoint locWirePos(layer->specificTopology().wirePosition(digi.wire()), 0, 0);
  const GlobalPoint globWirePos = layer->toGlobal(locWirePos);
  
  return compute(layer, wireId, digi.time(), globWirePos, leftPoint, rightPoint, error, 1); 
}



// Second step: the same as 1st step
bool DTLinearDriftAlgo::compute(const DTLayer* layer,
				const DTRecHit1D& recHit1D,
				const float& angle,
				DTRecHit1D& newHit1D) const {
  newHit1D.setPositionAndError(recHit1D.localPosition(), recHit1D.localPositionError());
  return true;
}



// Third step.
bool DTLinearDriftAlgo::compute(const DTLayer* layer,
				const DTRecHit1D& recHit1D,
				const float& angle,
				const GlobalPoint& globPos, 
				DTRecHit1D& newHit1D) const {
  return compute(layer, recHit1D.wireId(), recHit1D.digiTime(), globPos, newHit1D, 3);
}



// Do the actual work.
bool DTLinearDriftAlgo::compute(const DTLayer* layer,
				const DTWireId& wireId,
				const float digiTime,
				const GlobalPoint& globPos, 
				LocalPoint& leftPoint,
				LocalPoint& rightPoint,
				LocalError& error,
				int step) const {
  // Subtract the offset to the digi time accordingly to the DTTTrigBaseSync concrete instance
  float driftTime = digiTime - theSync->offset(layer, wireId, globPos); 
  
  // check for out-of-time
  if (driftTime < minTime || driftTime > maxTime) {
    if (debug) cout << "[DTLinearDriftAlgo]*** Drift time out of window for in-time hits "
			      << driftTime << endl;

    if(step == 1) { //FIXME: protection against failure at 2nd and 3rd steps, must be checked!!!
      // Hits are interpreted as coming from out-of-time pile-up and recHit
      // is ignored.
      return false;
    }
  }

  // Small negative times interpreted as hits close to the wire.
  if (driftTime<0.) driftTime=0;

  // Compute the drift distance
  // SL 21-Dec-2006: Use specific Drift for MB1W1 (non fluxed chamber)
  float vd=vDrift;
  // if (wireId.wheel()==1 && wireId.station()==1) {
  //   vd=vDriftMB1W1;
  //   //cout << "Using Vd " << vd<< endl;
  // }

  float drift = driftTime * vd;

  // Get Wire position
  if(!layer->specificTopology().isWireValid(wireId.wire())) return false;
  LocalPoint locWirePos(layer->specificTopology().wirePosition(wireId.wire()), 0, 0);
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
	 << "       Step:           " << step << endl
	 << "       Digi time:      " << digiTime << endl
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
bool DTLinearDriftAlgo::compute(const DTLayer* layer,
				const DTWireId& wireId,
				const float digiTime,
				const GlobalPoint& globPos, 
				DTRecHit1D& newHit1D,
				int step) const {
  LocalPoint leftPoint;
  LocalPoint rightPoint;
  LocalError error;

  if(compute(layer, wireId, digiTime, globPos, leftPoint, rightPoint, error, step)) {
    // Set the position and the error of the rechit which is being updated
    switch(newHit1D.lrSide()) {
	
    case DTEnums::Left:
        {
          // Keep the original y position of newHit1D: for step==3, it's the
          // position along the wire. Needed for rotation alignment
          LocalPoint leftPoint3D(leftPoint.x(), newHit1D.localPosition().y(), leftPoint.z());
          newHit1D.setPositionAndError(leftPoint3D, error);
          break;
        }
	
    case DTEnums::Right:
        {
          // as above: 3d position
          LocalPoint rightPoint3D(rightPoint.x(), newHit1D.localPosition().y(), rightPoint.z());
          newHit1D.setPositionAndError(rightPoint3D, error);
          break;
        }
	
    default:
      throw cms::Exception("InvalidDTCellSide") << "[DTLinearDriftAlgo] Compute at Step "
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


float DTLinearDriftAlgo::vDrift;
//float DTLinearDriftAlgo::vDriftMB1W1;

  
float DTLinearDriftAlgo::hitResolution;

  
float DTLinearDriftAlgo::minTime;

  
float DTLinearDriftAlgo::maxTime;

  
bool DTLinearDriftAlgo::debug;
