/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/02/15 13:54:45 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */

#include "DTTTrigSyncTOFCorr.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"

using namespace std;



DTTTrigSyncTOFCorr::DTTTrigSyncTOFCorr(const edm::ParameterSet& config){
  theTTrig = config.getParameter<double>("tTrig"); // FIXME: Default was 500 ns
  theVPropWire = config.getParameter<double>("vPropWire"); // FIXME: Default was 24.4 cm/ns
  theTOFCorrType = config.getParameter<int>("tofCorrType"); // FIXME: Default was 1
  debug = config.getUntrackedParameter<bool>("debug");

}



DTTTrigSyncTOFCorr::~DTTTrigSyncTOFCorr(){}



double DTTTrigSyncTOFCorr::offset(const DTLayer* layer,
				  const DTWireId& wireId,
				  const GlobalPoint& globPos,
				  double& tTrig,
				  double& wirePropCorr,
				  double& tofCorr) {
  // Correction for the float to int conversion
  // (half a bin on average) in DTDigi constructor
  static const float f2i_convCorr = (25./64.); // ns
  //FIXME: This should be considered only if the Digi is constructed from a float....


 // The tTrig is taken from a parameter
  tTrig = theTTrig - f2i_convCorr;

  //Compute the time spent in signal propagation along wire.
  // NOTE: the FE is always at y<0
  float halfL     = layer->specificTopology().cellLenght()/2;
  float wireCoord = layer->surface().toLocal(globPos).y();
  float propgL    = halfL + wireCoord;
  wirePropCorr = propgL/theVPropWire;


  // Compute TOF correction treating it accordingly to
  // the tofCorrType card
  float flightToHit = globPos.mag();
  const float cSpeed = 29.9792458; // cm/ns
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
//     // Correction for TOF from the center of the chamber to hit position
//     DTChamber chamber; //FIXME: Fake, how to get the chamber from the layer?
//     double flightToChamber = chamber.surface().position().mag();
//     tofCorr = (flightToChamber-flightToHit)/cSpeed;
    break;
  }
  case 2: {
    // TOF from 3D center of the wire to hit position
    float flightToWire =
      layer->surface().toGlobal(LocalPoint(layer->specificTopology().wirePosition(wireId.wire()), 0., 0.)).mag();
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
    cout << "[DTTTrigSyncTOFCorr] Offset: " << tTrig + wirePropCorr - tofCorr << endl
	 << "      various contributions are: " << endl
	 << "      tTrig:   " << tTrig << endl
	 << "      Popagation along wire delay: " <<  wirePropCorr << endl
	 << "      TOF: " << tofCorr << endl
	 << endl;
  }

  //The global offset is the sum of various contributions
  return tTrig + wirePropCorr - tofCorr;
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




