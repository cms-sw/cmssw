/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/03/14 13:02:42 $
 *  $Revision: 1.3 $
 *  \author G. Cerminara - INFN Torino
 */

#include "DTTTrigSyncFromDB.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
// #include "Geometry/DTGeometry/interface/DTLayer.h"
// #include "Geometry/DTGeometry/interface/DTChamber.h"
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
}



DTTTrigSyncFromDB::~DTTTrigSyncFromDB(){}



void DTTTrigSyncFromDB::setES(const EventSetup& setup) {
  ESHandle<DTT0> t0;
  setup.get<DTT0Rcd>().get(t0);
  tZeroMap = &*t0;
  
  ESHandle<DTTtrig> ttrig;
  setup.get<DTTtrigRcd>().get(ttrig);
  tTrigMap = &*ttrig;
  

  if(debug) {
    cout << "[DTTTrigSyncFromDB] T0 version: " << t0->version() << endl;
  }
}




double DTTTrigSyncFromDB::offset(const DTLayer* layer,
				  const DTWireId& wireId,
				  const GlobalPoint& globPos,
				  double& tTrig,
				  double& wirePropCorr,
				  double& tofCorr) {
  int t0 = 0; //FIXME: should become float
  float t0rms = 0;
  cout << "DEBUG: WireId: " << wireId << endl;
  tZeroMap->cellT0(2,
		   wireId.station(),
		   wireId.sector(),
		   wireId.superlayer(),
		   wireId.layer(),
		   wireId.wire(),
		   t0,
		   t0rms);
  cout << "DEBUG: t0: " << t0 << endl;

  int tt = 0;
  tTrigMap->slTtrig(2,
		    wireId.station(),
		    wireId.sector(),
		    wireId.superlayer(),
		    tt);
   cout << "DEBUG: tt: " << tt << endl;
  // Convert from tdc counts to ns
  tTrig = t0 * 25./32. + tt; //FIXME: move to ns






  wirePropCorr = 0;
  tofCorr = 0;

  if(debug) {
    cout << "[DTTTrigSyncFromDB] Offset (ns): " << tTrig + wirePropCorr - tofCorr << endl
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
bool DTTTrigSyncFromDB::debug;




