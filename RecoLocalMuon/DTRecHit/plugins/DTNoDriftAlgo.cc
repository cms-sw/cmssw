/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/04/30 09:30:06 $
 *  $Revision: 1.2 $
 *  \author Martijn Mulders - CERN (martijn.mulders@cern.ch)
 *  based on DTLinearDriftAlgo
 */

#include "RecoLocalMuon/DTRecHit/plugins/DTNoDriftAlgo.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;
using namespace edm;

DTNoDriftAlgo::DTNoDriftAlgo(const ParameterSet& config) :
  DTRecHitBaseAlgo(config) {

    minTime = config.getParameter<double>("minTime");

    maxTime = config.getParameter<double>("maxTime"); 

    fixedDrift = config.getParameter<double>("fixedDrift");

    hitResolution = config.getParameter<double>("hitResolution"); // Set to size of (half)cell 
    // Set verbose output
    debug = config.getUntrackedParameter<bool>("debug");
    
  }



DTNoDriftAlgo::~DTNoDriftAlgo(){}



void DTNoDriftAlgo::setES(const EventSetup& setup) {
  //  theSync->setES(setup);
}




// Build all hits in the range associated to the layerId, at the 1st step.
OwnVector<DTRecHit1DPair> DTNoDriftAlgo::reconstruct(const DTLayer* layer,
							const DTLayerId& layerId,
							const DTDigiCollection::Range& digiRange) {
  OwnVector<DTRecHit1DPair> result; 

  // Loop over all digis in the given range
  for (DTDigiCollection::const_iterator digi = digiRange.first;
       digi != digiRange.second;
       digi++) {
    // Get the wireId
    DTWireId wireId(layerId, (*digi).wire());

    bool isDouble = false;
    for (OwnVector<DTRecHit1DPair>::const_iterator doubleWireCheck =  result.begin();
         doubleWireCheck != result.end();
	 doubleWireCheck++) {
      if( wireId == (*doubleWireCheck).wireId()) {
	isDouble = true;
	//	std::cout << " Reject this hit with time " << (*digi).time() << std::endl; 
	break;
      }
    }
    
    if (isDouble) continue;

    LocalError tmpErr;
    LocalPoint lpoint, rpoint;
    // Call the compute method
    bool OK = compute(layer, *digi, lpoint, rpoint, tmpErr);

    if (!OK) continue;

    // Build a new pair of 1D rechit    
    DTRecHit1DPair*  recHitPair = new DTRecHit1DPair(wireId, *digi);

    // Set the position and the error of the 1D rechits
    recHitPair->setPositionAndError(DTEnums::Left, lpoint, tmpErr);
    recHitPair->setPositionAndError(DTEnums::Right, rpoint, tmpErr);        

    result.push_back(recHitPair);
  }
  return result;
}





// First Step
bool DTNoDriftAlgo::compute(const DTLayer* layer,
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
bool DTNoDriftAlgo::compute(const DTLayer* layer,
				const DTRecHit1D& recHit1D,
				const float& angle,
				DTRecHit1D& newHit1D) const {
  newHit1D.setPositionAndError(recHit1D.localPosition(), recHit1D.localPositionError());
  return true;
}



// Third step.
bool DTNoDriftAlgo::compute(const DTLayer* layer,
				const DTRecHit1D& recHit1D,
				const float& angle,
				const GlobalPoint& globPos, 
				DTRecHit1D& newHit1D) const {
  return compute(layer, recHit1D.wireId(), recHit1D.digiTime(), globPos, newHit1D, 3);
}



// Do the actual work.
bool DTNoDriftAlgo::compute(const DTLayer* layer,
				const DTWireId& wireId,
				const float digiTime,
				const GlobalPoint& globPos, 
				LocalPoint& leftPoint,
				LocalPoint& rightPoint,
				LocalError& error,
				int step) const {
  //}

  // Small negative times interpreted as hits close to the wire.
  //if (driftTime<0.) driftTime=0;


  // check for out-of-time
  if (digiTime < minTime || digiTime > maxTime) {
    if (debug) cout << "[DTNoDriftAlgo]*** Drift time out of window for in-time hits "
			      << digiTime << endl;

    if(step == 1) { //FIXME: protection against failure at 2nd and 3rd steps, must be checked!!!
      // Hits are interpreted as coming from out-of-time pile-up and recHit
      // is ignored.
      return false;
    }
  }


  // Compute the drift distance
  float drift = fixedDrift;

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
    cout << "[DTNoDriftAlgo] Compute drift distance, for digi at wire: " << wireId << endl
	 << "       Step:           " << step << endl
      	 << "       Digi time:      " << digiTime << endl
      //	 << "       Drift time:     " << driftTime << endl
      	 << "       Fixed Drift distance: " << drift << endl
	 << "       Hit Resolution: " << hitResolution << endl
	 << "       Left point:     " << leftPoint << endl
	 << "       Right point:    " << rightPoint << endl
	 << "       Error:          " << error << endl;
  }



  return true;
  
}


// Interface to the method which does the actual work suited for 2nd and 3rd steps 
bool DTNoDriftAlgo::compute(const DTLayer* layer,
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
      newHit1D.setPositionAndError(leftPoint, error);
      break;
	
    case DTEnums::Right:
      newHit1D.setPositionAndError(rightPoint, error);
      break;
	
    default:
      throw cms::Exception("InvalidDTCellSide") << "[DTNoDriftAlgo] Compute at Step "
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


float DTNoDriftAlgo::fixedDrift;

  
float DTNoDriftAlgo::hitResolution;

  
float DTNoDriftAlgo::minTime;

  
float DTNoDriftAlgo::maxTime;

  
bool DTNoDriftAlgo::debug;
