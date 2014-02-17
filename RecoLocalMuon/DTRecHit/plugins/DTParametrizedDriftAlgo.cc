/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/04/30 09:30:07 $
 *  $Revision: 1.3 $
 *  \author G. Cerminara - INFN Torino
 */

#include "RecoLocalMuon/DTRecHit/plugins/DTParametrizedDriftAlgo.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"
#include "RecoLocalMuon/DTRecHit/plugins/DTTime2DriftParametrization.h"

#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <iostream>
#include <iomanip>

using namespace std;
using namespace edm;


DTParametrizedDriftAlgo::DTParametrizedDriftAlgo(const ParameterSet& config) :
  DTRecHitBaseAlgo(config) {
    interpolate = config.getParameter<bool>("interpolate");

    minTime = config.getParameter<double>("minTime"); // FIXME: Default was -3 ns

    maxTime = config.getParameter<double>("maxTime"); // FIXME: Default was 415 ns

    // Set verbose output
    debug = config.getUntrackedParameter<bool>("debug","false");
    
  }



DTParametrizedDriftAlgo::~DTParametrizedDriftAlgo(){}



void DTParametrizedDriftAlgo::setES(const EventSetup& setup) {
  theSync->setES(setup);
  // Access the magnetic field
  ESHandle<MagneticField> magneticField;
  setup.get<IdealMagneticFieldRecord>().get(magneticField);
  magField = &*magneticField;
}



// First Step
bool DTParametrizedDriftAlgo::compute(const DTLayer* layer,
				      const DTDigi& digi,
				      LocalPoint& leftPoint,
				      LocalPoint& rightPoint,
				      LocalError& error) const {
  // Get the layerId
  DTLayerId layerId = layer->id();//FIXME: pass it instead of get it from layer
  const DTWireId wireId(layerId, digi.wire());
  
  // Get Wire position
  if(!layer->specificTopology().isWireValid(wireId.wire())) return false;
  LocalPoint locWirePos(layer->specificTopology().wirePosition(wireId.wire()), 0, 0);
  const GlobalPoint globWirePos = layer->toGlobal(locWirePos);
  
  // impact angle on DT chamber
  // compute the impact angle using the centre of the wire,
  // only for RZ superlayer (eta). In the other cases theta is very close to 0.
  float angle=0.0;    

  if (layerId.superlayer() == 2 ) {
    //GlobalPoint lPos=layer->position();
    GlobalVector lDir=(GlobalPoint()-globWirePos).unit();
    LocalVector lDirLoc=layer->toLocal(lDir);

    angle = atan(lDirLoc.x()/-lDirLoc.z());
  } 
  
  return compute(layer, wireId, digi.time(), angle, globWirePos,
                 leftPoint, rightPoint, error, 1);
}



// Second step
bool DTParametrizedDriftAlgo::compute(const DTLayer* layer,
				      const DTRecHit1D& recHit1D,
				      const float& angle,
				      DTRecHit1D& newHit1D) const {
  const DTWireId wireId = recHit1D.wireId();
  
  // Get Wire position
  if(!layer->specificTopology().isWireValid(wireId.wire())) return false;
  LocalPoint locWirePos(layer->specificTopology().wirePosition(wireId.wire()), 0, 0);
  const GlobalPoint globWirePos = layer->toGlobal(locWirePos);

  return compute(layer, wireId, recHit1D.digiTime(), angle, globWirePos,
                 newHit1D, 2);
}



// Third step
bool DTParametrizedDriftAlgo::compute(const DTLayer* layer,
				      const DTRecHit1D& recHit1D,
				      const float& angle,
				      const GlobalPoint& globPos, 
				      DTRecHit1D& newHit1D) const {
  return compute(layer, recHit1D.wireId(), recHit1D.digiTime(), angle,
		 globPos, newHit1D, 3);
}



// Do the actual work.
bool DTParametrizedDriftAlgo::compute(const DTLayer* layer,
				      const DTWireId& wireId,
				      const float digiTime,
				      const float& angle,
				      const GlobalPoint& globPos, 
				      LocalPoint& leftPoint,
				      LocalPoint& rightPoint,
				      LocalError& error,
				      int step) const {
  // Subtract Offset according to DTTTrigBaseSync concrete instance
  // chosen with the 'tZeroMode' parameter
  float driftTime = digiTime - theSync->offset(layer, wireId, globPos);

  // check for out-of-time only at step 1
  if (step==1 && (driftTime < minTime || driftTime > maxTime)) {
    if (debug)
      cout << "*** Drift time out of window for in-time hits "
	   << driftTime << endl;

    if(step == 1) { //FIXME: protection against failure at 2nd and 3rd steps, must be checked!!!
      // Hits are interpreted as coming from out-of-time pile-up and recHit
      // is ignored.
      return false;
    }
  }
  
  // Small negative times interpreted as hits close to the wire.
  if (driftTime<0.) driftTime=0;

  //----------------------------------------------------------------------
  // Magnetic Field in layer frame
  const LocalVector BLoc =
    layer->toLocal(magField->inTesla(globPos));

  float By = BLoc.y();
  float Bz = BLoc.z();  

  //--------------------------------------------------------------------
  // Calculate the drift distance and the resolution from the parametrization
  
  DTTime2DriftParametrization::drift_distance DX;
  static DTTime2DriftParametrization par;

  bool parStatus =
    par.computeDriftDistance_mean(driftTime, angle, By, Bz, interpolate, &DX);

  if (!parStatus) {
    if (debug)
      cout << "[DTParametrizedDriftAlgo]*** WARNING: call to parametrization failed" << endl;
    return false;
  }

  float sigma_l = DX.x_width_m;
  float sigma_r = DX.x_width_p;
  float drift = DX.x_drift;


  float reso = 0;

  bool variableSigma = false;
  // By defualt the errors are obtained from a fit of the residuals in the various
  // stations/SL.
  // The error returned by DTTime2DriftParametrization can not be used easily
  // to determine the hit error due to the way the parametrization of the error
  // is done (please contcat the authors for details).
  // Anyhow changing variableSigma==true, an attempt is done to set a variable error
  // according to the sigma calculated by DTTime2DriftParametrization.
  // Additionally, contributions from uncertaionties in TOF and signal propagation
  // corrections are added.
  // Uncertainty in the determination of incident angle and hit position
  // (ie B value) are NOT accounted.
  // This is not the default since it does not give good results...

  int wheel = abs(wireId.wheel());
  if (variableSigma) {
    float sTDelays=0;
    if (step==1) {               // 1. step
      reso = (sigma_l+sigma_r)/2.; // FIXME: theta/B are not yet known...
      if (wireId.superlayer()==2) {    // RZ
        sTDelays = 2.92;
      } else {                   // RPhi
        if (wheel==0) {
          sTDelays = 2.74;
        } else if (wheel==1) {
          sTDelays = 1.83;
        } else if (wheel==2){
          sTDelays = 1.25;
        }
      } 
    } else if (step==2) {        // 2. step
      reso = (sigma_l+sigma_r)/2.; // FIXME: B is not yet known...
      if (wireId.superlayer()==2) {    // RZ
        sTDelays = 0.096;
      } else {                   // RPhi
        if (wheel==0) {
          sTDelays = 0.022;
        } else if (wheel==1) {
          sTDelays = 0.079;
        } else if (wheel==2){
          sTDelays = 0.124;
        }
      }      
    } else if (step==3) {        // 3. step
      reso = (sigma_l+sigma_r)/2.;
      if (wireId.superlayer()==2) {    // RZ
        sTDelays = 0.096;
      } else {                   // RPhi
        if (wheel==0) {
          sTDelays = 0.022;
        } else if (wheel==1) {
          sTDelays = 0.079;
        } else if (wheel==2){
          sTDelays = 0.124;
        }
      }
    }
    float sXDelays = sTDelays*DX.v_drift/10.; 
    reso = sqrt(reso*reso + sXDelays*sXDelays);
  } else { // Use a fixed sigma, from fit of residuals.
    if (step==1) {     // 1. step
      if (wireId.superlayer()==2) {     
        if (wheel==0) {
          reso = 0.0250;
        } else if (wheel==1) {
          reso = 0.0271;
        } else if (wheel==2){
          reso = 0.0308;
        }
      } else {
        reso = 0.0237;
      }
    } else if (step==2) {                  // 2. step //FIXME
      if (wireId.superlayer()==2) {
        if (wheel==0) {
          reso = 0.0250;
        } else if (wheel==1) {
          reso = 0.0271;
        } else if (wheel==2){
          reso = 0.0305;
        }
      } else {
        reso = 0.0231;
      }
    } else if (step==3) {                  // 3. step
      if (wireId.superlayer()==2) {
        if (wheel==0) {
          reso = 0.0196;
        } else if (wheel==1) {
          reso = 0.0210;
        } else if (wheel==2){
          reso = 0.0228;
        }
      } else {
        reso = 0.0207;
      }
    }
  }
  //--------------------------------------------------------------------

  error = LocalError(reso*reso,0.,0.);

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

  if (debug) {
    int prevW = cout.width();
    cout << setiosflags(ios::showpoint | ios::fixed) << setw(3)
         << "[DTParametrizedDriftAlgo]: step " << step << endl
         << "  Global Position  " << globPos << endl
         << "  Local Position   " << layer->toLocal(globPos) << endl
      //          << "  y along Wire     " << wireCoord << endl
         << "  Digi time        " << digiTime << endl
      //          << "  dpropDelay       " << propDelay << endl
      //          << "  deltaTOF         " << deltaTOF << endl
         << " >Drif time        " << driftTime << endl
         << " >Angle            " << angle * 180./M_PI << endl
         << " <Drift distance   " << drift << endl
	 << " <sigma_l          " << sigma_l << endl
	 << " <sigma_r          " << sigma_r << endl
         << "  reso             " << reso << endl
         << "  leftPoint        " << leftPoint << endl
         << "  rightPoint       " << rightPoint << endl
         << "  error            " << error
	 <<  resetiosflags(ios::showpoint | ios::fixed)  << setw(prevW) << endl 
         << endl;  
  }

  return true;
}


// Interface to the method which does the actual work suited for 2nd and 3rd steps 
bool DTParametrizedDriftAlgo::compute(const DTLayer* layer,
				      const DTWireId& wireId,
				      const float digiTime,
				      const float& angle,
				      const GlobalPoint& globPos, 
				      DTRecHit1D& newHit1D,
				      int step) const {
  LocalPoint leftPoint;
  LocalPoint rightPoint;
  LocalError error;
    
  if(compute(layer, wireId, digiTime, angle, globPos, leftPoint, rightPoint, error, step)) {
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
      throw cms::Exception("InvalidDTCellSide") << "[DTParametrizedDriftAlgo] Compute at Step "
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


bool DTParametrizedDriftAlgo::interpolate;


float DTParametrizedDriftAlgo::minTime;

  
float DTParametrizedDriftAlgo::maxTime;

  
bool DTParametrizedDriftAlgo::debug;
