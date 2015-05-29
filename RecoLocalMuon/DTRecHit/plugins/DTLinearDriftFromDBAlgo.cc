/*
 *  See header file for a description of this class.
 *
 *  \author S. Bolognesi - INFN Torino
 */

#include "RecoLocalMuon/DTRecHit/plugins/DTLinearDriftFromDBAlgo.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CondFormats/DTObjects/interface/DTMtime.h"
#include "CondFormats/DataRecord/interface/DTMtimeRcd.h"
#include "CondFormats/DTObjects/interface/DTRecoUncertainties.h"
#include "CondFormats/DataRecord/interface/DTRecoUncertaintiesRcd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

using namespace std;
using namespace edm;

DTLinearDriftFromDBAlgo::DTLinearDriftFromDBAlgo(const ParameterSet& config) :
  DTRecHitBaseAlgo(config),
  mTimeMap(0),
  field(0),
  nominalB(-1),
  minTime(config.getParameter<double>("minTime")),
  maxTime(config.getParameter<double>("maxTime")),
  doVdriftCorr(config.getParameter<bool>("doVdriftCorr")),
  // Option to force going back to digi time at Step 2 
  stepTwoFromDigi(config.getParameter<bool>("stepTwoFromDigi")),
  useUncertDB(false),
  // Set verbose output
  debug(config.getUntrackedParameter<bool>("debug"))
{
  if(debug)
    cout<<"[DTLinearDriftFromDBAlgo] Constructor called"<<endl;

  // Check for compatibility with older configurations
  if (config.existsAs<bool>("useUncertDB")) {
    // Assign hit uncertainties based on new uncertainties DB
    useUncertDB= config.getParameter<bool>("useUncertDB");
  }
}



DTLinearDriftFromDBAlgo::~DTLinearDriftFromDBAlgo(){}



void DTLinearDriftFromDBAlgo::setES(const EventSetup& setup) {
  if(debug)
    cout<<"[DTLinearDriftFromDBAlgo] setES called"<<endl;
  theSync->setES(setup);
  // Get the map of ttrig from the Setup
  ESHandle<DTMtime> mTimeHandle;
  setup.get<DTMtimeRcd>().get(mTimeHandle);
  mTimeMap = &*mTimeHandle;

  ESHandle<MagneticField> magfield;
  setup.get<IdealMagneticFieldRecord>().get(magfield);
  field = &*magfield;
  nominalB = field->nominalValue();

  if (useUncertDB) {
    ESHandle<DTRecoUncertainties> uncerts;
    setup.get<DTRecoUncertaintiesRcd>().get(uncerts);
    uncertMap = &*uncerts;
  
    // check uncertainty map type
    if (uncertMap->version()>1) edm::LogError("NotImplemented") << "DT Uncertainty DB version unknown: " << uncertMap->version();
  }
  
  if(debug) {
    cout << "[DTLinearDriftFromDBAlgo] meanTimer version: " << mTimeMap->version()<<endl;
    if (useUncertDB) cout << "                          uncertDB  version: " << uncertMap->version()<<endl;
  }
  
}



// First Step
bool DTLinearDriftFromDBAlgo::compute(const DTLayer* layer,
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



// Second step: the same as 1st step (optionally, redo 1st step starting from digi time)
bool DTLinearDriftFromDBAlgo::compute(const DTLayer* layer,
				const DTRecHit1D& recHit1D,
				const float& angle,
				DTRecHit1D& newHit1D) const {

  if (!stepTwoFromDigi) {
    newHit1D.setPositionAndError(recHit1D.localPosition(), recHit1D.localPositionError());
    return true;
  }

  const DTWireId wireId = recHit1D.wireId();
  
  // Get Wire position
  if(!layer->specificTopology().isWireValid(wireId.wire())) return false;
  LocalPoint locWirePos(layer->specificTopology().wirePosition(wireId.wire()), 0, 0);
  const GlobalPoint globWirePos = layer->toGlobal(locWirePos);

  return compute(layer, wireId, recHit1D.digiTime(), globWirePos, newHit1D, 2);

}



// Third step.
bool DTLinearDriftFromDBAlgo::compute(const DTLayer* layer,
				const DTRecHit1D& recHit1D,
				const float& angle,
				const GlobalPoint& globPos, 
				DTRecHit1D& newHit1D) const {
  return compute(layer, recHit1D.wireId(), recHit1D.digiTime(), globPos, newHit1D, 3);
}



// Do the actual work.
bool DTLinearDriftFromDBAlgo::compute(const DTLayer* layer,
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
    if (debug) cout << "[DTLinearDriftFromDBAlgo]*** Drift time out of window for in-time hits "
			      << driftTime << endl;

    if(step == 1) { //FIXME: protection against failure at 2nd and 3rd steps, must be checked!!!
      // Hits are interpreted as coming from out-of-time pile-up and recHit
      // is ignored.
      return false;
    }
  }

  // Small negative times interpreted as hits close to the wire.
  if (driftTime<0.) driftTime=0;

  // Read the vDrift and reso for this wire
  float vDrift = 0;
  float hitResolution = 0; 
  // vdrift is cm/ns , resolution is cm
  mTimeMap->get(wireId.superlayerId(),
	        vDrift,
	        hitResolution,  // Value from vdrift DB; replaced below if useUncertDB card is set
	        DTVelocityUnits::cm_per_ns);

  if (useUncertDB) {
    // Read the uncertainty from the DB for the given channel and step
    hitResolution = uncertMap->get(wireId, step-1);
  }
  
  //only in step 3
  if(doVdriftCorr && step == 3 && nominalB !=0){
    if (abs(wireId.wheel()) == 2 && 
	wireId.station() == 1 &&
	wireId.superLayer() != 2) {
      // Variation of vdrift along Y due to B field, 
      /// cf. http://arxiv.org/PS_cache/arxiv/pdf/0911/0911.4895v2.pdf
      // vdrift is lower  a negative Y (lower global |Z|)
      const float k_param = 1.2e-04;
      LocalPoint local_pos = layer->toLocal(globPos);
      vDrift = vDrift*(1. - k_param*local_pos.y());
	} 
  }

  // Compute the drift distance
  float drift = driftTime * vDrift;

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
    cout << "[DTLinearDriftFromDBAlgo] Compute drift distance, for digi at wire: " << wireId << endl
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
bool DTLinearDriftFromDBAlgo::compute(const DTLayer* layer,
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
      throw cms::Exception("InvalidDTCellSide") << "[DTLinearDriftFromDBAlgo] Compute at Step "
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
