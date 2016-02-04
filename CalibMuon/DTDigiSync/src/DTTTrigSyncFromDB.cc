/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/12/07 17:22:18 $
 *  $Revision: 1.10 $
 *  \author G. Cerminara - INFN Torino
 */

#include "DTTTrigSyncFromDB.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DataRecord/interface/DTT0Rcd.h"
#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"

#include <iostream>

using namespace std;
using namespace edm;


DTTTrigSyncFromDB::DTTTrigSyncFromDB(const ParameterSet& config){
  debug = config.getUntrackedParameter<bool>("debug");
  // The velocity of signal propagation along the wire (cm/ns)
  theVPropWire = config.getParameter<double>("vPropWire");
  // Switch on/off the T0 correction from pulses
  doT0Correction = config.getParameter<bool>("doT0Correction");
  // Switch on/off the TOF correction for particles from IP
  doTOFCorrection = config.getParameter<bool>("doTOFCorrection");
  theTOFCorrType = config.getParameter<int>("tofCorrType");
  // Switch on/off the correction for the signal propagation along the wire
  doWirePropCorrection = config.getParameter<bool>("doWirePropCorrection");
  theWirePropCorrType = config.getParameter<int>("wirePropCorrType");
  // spacing of BX in ns
  theBXspace  = config.getUntrackedParameter<double>("bxSpace", 25.);

  thetTrigLabel = config.getParameter<string>("tTrigLabel");
}



DTTTrigSyncFromDB::~DTTTrigSyncFromDB(){}



void DTTTrigSyncFromDB::setES(const EventSetup& setup) {
  if(doT0Correction)
    {
      // Get the map of t0 from pulses from the Setup
      ESHandle<DTT0> t0Handle;
      setup.get<DTT0Rcd>().get(t0Handle);
      tZeroMap = &*t0Handle;
      if(debug) {
	cout << "[DTTTrigSyncFromDB] t0 version: " << tZeroMap->version()<<endl;
	  }
    }

  // Get the map of ttrig from the Setup
  ESHandle<DTTtrig> ttrigHandle;
  setup.get<DTTtrigRcd>().get(thetTrigLabel,ttrigHandle);
  tTrigMap = &*ttrigHandle;
    if(debug) {
      cout << "[DTTTrigSyncFromDB] ttrig version: " << tTrigMap->version() << endl;
  }
}


double DTTTrigSyncFromDB::offset(const DTLayer* layer,
				  const DTWireId& wireId,
				  const GlobalPoint& globPos,
				  double& tTrig,
				  double& wirePropCorr,
				  double& tofCorr) {
  // Correction for the float to int conversion while writeing the ttrig in ns into an int variable
  // (half a bin on average)
  // FIXME: this should disappear as soon as the ttrig object will become a float
  //   static const float f2i_convCorr = (25./64.); // ns //FIXME: check how the conversion is performed

  tTrig = offset(wireId);

  // Compute the time spent in signal propagation along wire.
   // NOTE: the FE is always at y>0
  wirePropCorr = 0;
  if(doWirePropCorrection) {
    switch(theWirePropCorrType){
      // The ttrig computed from the timebox accounts on average for the signal propagation time
      // from the center of the wire to the frontend. Here we just have to correct for
      // the distance of the hit from the wire center.
    case 0: {
      float wireCoord = layer->toLocal(globPos).y();
      wirePropCorr = -wireCoord/theVPropWire;
      break;
      // FIXME: What if hits used for the time box are not distributed uniformly along the wire?
    }
    //On simulated data you need to subtract the total propagation time 
    case 1: {
      float halfL     = layer->specificTopology().cellLenght()/2;
      float wireCoord = layer->toLocal(globPos).y();
      float propgL    = halfL - wireCoord;
      wirePropCorr = propgL/theVPropWire;
      break;
    }
    default: {
    throw
      cms::Exception("[DTTTrigSyncFromDB]") << " Invalid parameter: wirePropCorrType = "
					     << theWirePropCorrType 
					     << std::endl;
    break;
    }
    }
  }

  // Compute TOF correction:
  tofCorr = 0.;
  // TOF Correction can be switched off with appropriate parameter
  if(doTOFCorrection) {
    float flightToHit = globPos.mag();
    static const float cSpeed = 29.9792458; // cm/ns
    switch(theTOFCorrType) {
    case 0: {
      // The ttrig computed from the real data accounts on average for the TOF correction 
      // Depending on the granularity used for the ttrig computation we just have to correct for the
      // TOF from the center of the chamber, SL, layer or wire to the hit position.
      // At the moment only SL granularity is considered
      // Correction for TOF from the center of the SL to hit position
      const DTSuperLayer *sl = layer->superLayer();
      double flightToSL = sl->surface().position().mag();
      tofCorr = (flightToSL-flightToHit)/cSpeed;
      break;
    }
    case 1: {
      // On simulated data you need to consider only the TOF from 3D center of the wire to hit position 
      // (because the TOF from the IP to the wire has been already subtracted in the digitization: 
      // SimMuon/DTDigitizer/DTDigiSyncTOFCorr.cc corrType=2)
      float flightToWire =
	layer->toGlobal(LocalPoint(layer->specificTopology().wirePosition(wireId.wire()), 0., 0.)).mag();
      tofCorr = (flightToWire-flightToHit)/cSpeed;
      break;
    }
    default: {
      throw
	cms::Exception("[DTTTrigSyncFromDB]") << " Invalid parameter: tofCorrType = "
					      << theTOFCorrType 
					      << std::endl;
      break;
    }
    }
  }


  if(debug) {
    cout << "[DTTTrigSyncFromDB] Channel: " << wireId << endl
	 << "      Offset (ns): " << tTrig + wirePropCorr - tofCorr << endl
	 << "      various contributions are: " << endl
	 << "      tTrig + t0 (ns):   " << tTrig << endl
	 //<< "      tZero (ns):   " << t0 << endl
	 << "      Propagation along wire delay (ns): " <<  wirePropCorr << endl
	 << "      TOF correction (ns): " << tofCorr << endl
	 << endl;
  }
  //The global offset is the sum of various contributions
    return tTrig + wirePropCorr - tofCorr;
}

double DTTTrigSyncFromDB::offset(const DTWireId& wireId) {
  float t0 = 0;
  float t0rms = 0;
  if(doT0Correction)
    {
      // Read the t0 from pulses for this wire (ns)
       tZeroMap->get(wireId,
		     t0,
		     t0rms,
	             DTTimeUnits::ns);

    }

  // Read the ttrig for this wire
  float ttrigMean = 0;
  float ttrigSigma = 0;
  float kFactor = 0;
  // FIXME: should check the return value of the DTTtrigRcd::get(..) method
  if(tTrigMap->get(wireId.superlayerId(),
		   ttrigMean,
		   ttrigSigma,
		   kFactor,
		   DTTimeUnits::ns) != 0) {
    cout << "[DTTTrigSyncFromDB]*Error: ttrig not found for SL: " << wireId.superlayerId() << endl;
//     FIXME: LogError.....
  }
  
  return t0 + ttrigMean + kFactor * ttrigSigma;

}


// Set the verbosity level
bool DTTTrigSyncFromDB::debug = false;


double DTTTrigSyncFromDB::emulatorOffset(const DTWireId& wireId,
					 double &tTrig,
					 double &t0cell) {
  float t0 = 0;
  float t0rms = 0;
  if(doT0Correction)
    {
      // Read the t0 from pulses for this wire (ns)
       tZeroMap->get(wireId,
		     t0,
		     t0rms,
	             DTTimeUnits::ns);

    }

  // Read the ttrig for this wire
  float ttrigMean = 0;
  float ttrigSigma = 0;
  float kFactor = 0;
  // FIXME: should check the return value of the DTTtrigRcd::get(..) method
  if(tTrigMap->get(wireId.superlayerId(),
		   ttrigMean,
		   ttrigSigma,
		   kFactor,
		   DTTimeUnits::ns) != 0) {
    cout << "[DTTTrigSyncFromDB]*Error: ttrig not found for SL: " << wireId.superlayerId() << endl;
//     FIXME: LogError.....
  }
  
  tTrig = ttrigMean + kFactor * ttrigSigma;
  t0cell = t0;

  return int(tTrig/theBXspace)*theBXspace + t0cell;
}
