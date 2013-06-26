/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/10/31 15:02:21 $
 *  $Revision: 1.6 $
 *  \author G. Cerminara - INFN Torino
 */

#include "DTTTrigSyncTOFCorr.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include <iostream>

using namespace std;



DTTTrigSyncTOFCorr::DTTTrigSyncTOFCorr(const edm::ParameterSet& config){
  theTTrig = config.getParameter<double>("tTrig"); // FIXME: Default was 500 ns
  theVPropWire = config.getParameter<double>("vPropWire"); // FIXME: Default was 24.4 cm/ns
  theTOFCorrType = config.getParameter<int>("tofCorrType"); // FIXME: Default was 1
  debug = config.getUntrackedParameter<bool>("debug");
  // spacing of BX in ns
  theBXspace  = config.getUntrackedParameter<double>("bxSpace", 25.);

}



DTTTrigSyncTOFCorr::~DTTTrigSyncTOFCorr(){}



double DTTTrigSyncTOFCorr::offset(const DTLayer* layer,
				  const DTWireId& wireId,
				  const GlobalPoint& globPos,
				  double& tTrig,
				  double& wirePropCorr,
				  double& tofCorr) {
  tTrig = offset(wireId);

  //Compute the time spent in signal propagation along wire.
  // NOTE: the FE is always at y>0
  float halfL     = layer->specificTopology().cellLenght()/2;
  float wireCoord = layer->toLocal(globPos).y();
  float propgL    = halfL - wireCoord;
  wirePropCorr = propgL/theVPropWire;


  // Compute TOF correction treating it accordingly to
  // the tofCorrType card
  float flightToHit = globPos.mag();
  static const float cSpeed = 29.9792458; // cm/ns
  tofCorr = 0.;
  switch(theTOFCorrType) {
  case 0: {
    // In this mode the subtraction of the TOF from IP to
    // estimate 3D hit digi position is done here
    // (No correction is needed anymore)
    tofCorr = -flightToHit/cSpeed;
    break;
  }
  case 1: {
    // Correction for TOF from the center of the chamber to hit position
    const DTChamber * chamber = layer->chamber();
    double flightToChamber = chamber->surface().position().mag();
    tofCorr = (flightToChamber-flightToHit)/cSpeed;
    break;
  }
  case 2: {
    // TOF from 3D center of the wire to hit position
    float flightToWire =
      layer->toGlobal(LocalPoint(layer->specificTopology().wirePosition(wireId.wire()), 0., 0.)).mag();
    tofCorr = (flightToWire-flightToHit)/cSpeed;
    break;
  }
  default: {
    throw
      cms::Exception("[DTTTrigSyncTOFCorr]") << " Invalid parameter: tofCorrType = "
					     << theTOFCorrType 
					     << std::endl;
    break;
  }
  }

  if(debug) {
    cout << "[DTTTrigSyncTOFCorr] Offset (ns): " << tTrig + wirePropCorr - tofCorr << endl
	 << "      various contributions are: " << endl
	 << "      tTrig (ns):   " << tTrig << endl
	 << "      Propagation along wire delay (ns): " <<  wirePropCorr << endl
	 << "      TOF correction (ns): " << tofCorr << endl
	 << endl;
  }
  //The global offset is the sum of various contributions
  return tTrig + wirePropCorr - tofCorr;
}

double DTTTrigSyncTOFCorr::offset(const DTWireId& wireId) {
  // Correction for the float to int conversion
  // (half a bin on average) in DTDigi constructor
  static const float f2i_convCorr = (25./64.); // ns
  //FIXME: This should be considered only if the Digi is constructed from a float....


  // The tTrig is taken from a parameter
  return theTTrig - f2i_convCorr;
}


// The fixed t0 (or t_trig) to be subtracted to digi time (ns)
double DTTTrigSyncTOFCorr::theTTrig;



// Velocity of signal propagation along the wire (cm/ns)
double DTTTrigSyncTOFCorr::theVPropWire;


  
// Select the mode for TOF correction:
//     0: tofCorr = TOF from IP to 3D Hit position (globPos)
//     1: tofCorr = TOF correction for distance difference
//                  between 3D center of the chamber and hit position
//                  (default)
//     2: tofCorr = TOF correction for distance difference
//                  between 3D center of the wire and hit position
//                  (This mode in available for backward compatibility)
int DTTTrigSyncTOFCorr::theTOFCorrType;



// Set the verbosity level
bool DTTTrigSyncTOFCorr::debug;



double DTTTrigSyncTOFCorr::emulatorOffset(const DTWireId& wireId,
					 double &tTrig,
					 double &t0cell) {
  tTrig = theTTrig;
  t0cell = 0.;

  return int(tTrig/theBXspace)*theBXspace;
}

