/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/10/21 17:05:47 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara - INFN Torino
 */

#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"


DTTTrigBaseSync::DTTTrigBaseSync(){}



DTTTrigBaseSync::~DTTTrigBaseSync(){}



double DTTTrigBaseSync::offset(const DTLayer* layer,
			       const DTWireId& wireId,
			       const GlobalPoint& globalPos) {
  double tTrig = 0;
  double wireProp = 0;
  double tof = 0;
  return offset(layer, wireId, globalPos, tTrig, wireProp, tof);
}



double DTTTrigBaseSync::emulatorOffset(const DTWireId& wireId) {

  double tTrig = 0.;
  double t0cell = 0.;
  return emulatorOffset(wireId, tTrig, t0cell);
}
