/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/03/14 13:02:42 $
 *  $Revision: 1.3 $
 *  \author G. Cerminara - INFN Torino
 */

#include "DTTTrigSyncT0Only.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
// #include "Geometry/DTGeometry/interface/DTLayer.h"
// #include "Geometry/DTGeometry/interface/DTChamber.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DataRecord/interface/DTT0Rcd.h"

#include <iostream>

using namespace std;
using namespace edm;


DTTTrigSyncT0Only::DTTTrigSyncT0Only(const ParameterSet& config){
  debug = config.getUntrackedParameter<bool>("debug");
}



DTTTrigSyncT0Only::~DTTTrigSyncT0Only(){}



void DTTTrigSyncT0Only::setES(const EventSetup& setup) {
  ESHandle<DTT0> t0;
  setup.get<DTT0Rcd>().get(t0);
  tZeroMap = &*t0;
  
  if(debug) {
    cout << "[DTTTrigSyncT0Only] T0 version: " << t0->version() << endl;
  }
}




double DTTTrigSyncT0Only::offset(const DTLayer* layer,
				  const DTWireId& wireId,
				  const GlobalPoint& globPos,
				  double& tTrig,
				  double& wirePropCorr,
				  double& tofCorr) {
  int t0 = 0; //FIXME: should become float
  float t0rms = 0;
  tZeroMap->cellT0(wireId.wheel(),
		   wireId.station(),
		   wireId.sector(),
		   wireId.superlayer(),
		   wireId.layer(),
		   wireId.wire(),
		   t0,
		   t0rms);
  // Convert from tdc counts to ns
  tTrig = t0 * 25./32.; //FIXME: move to ns
  wirePropCorr = 0;
  tofCorr = 0;

  if(debug) {
    cout << "[DTTTrigSyncT0Only] Offset (ns): " << tTrig + wirePropCorr - tofCorr << endl
	 << "      various contributions are: " << endl
	 << "      tTrig (ns):   " << tTrig << endl
	 << "      Propagation along wire delay (ns): " <<  wirePropCorr << endl
	 << "      TOF correction (ns): " << tofCorr << endl
	 << endl;
  }
  //The global offset is the sum of various contributions
  return tTrig + wirePropCorr - tofCorr;
}



// Set the verbosity level
bool DTTTrigSyncT0Only::debug;




