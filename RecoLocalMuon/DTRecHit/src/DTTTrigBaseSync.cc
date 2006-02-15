/*
 *  See header file for a description of this class.
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara - INFN Torino
 */

#include "RecoLocalMuon/DTRecHit/interface/DTTTrigBaseSync.h"


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

