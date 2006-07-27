#include "L1Trigger/RPCTrigger/src/L1RpcLogHit.h"


/**
 *
 *\brief Default ctor
 *\todo Check LogSector/LogSegment convention
 *
*/
L1RpcLogHit::L1RpcLogHit(int tower, int PAC, int logplane, int posInCone){

  ConeCrdnts.Tower=tower;
  ConeCrdnts.LogSector=PAC/12;
  ConeCrdnts.LogSegment=PAC%12;
  stripNumberInCone = posInCone;
  logPlaneNumber = logplane;
  
}
//###################################################################
//
// Simple getters and setters
//
//###################################################################
L1RpcConst::L1RpcConeCrdnts L1RpcLogHit::GetConeCrdnts() const {
  return ConeCrdnts;
}

int L1RpcLogHit::getTower() const {
  return ConeCrdnts.Tower;
}

int L1RpcLogHit::getLogSector() const {
  return ConeCrdnts.LogSector;
}

int L1RpcLogHit::getLogSegment() const {
  return ConeCrdnts.LogSegment;
}

int L1RpcLogHit::getlogPlaneNumber() const {
  return logPlaneNumber;
}

int L1RpcLogHit::getStripNumberInCone() const {
  return stripNumberInCone;
}

void  L1RpcLogHit::setDigiIdx(int _digiIdx) {
  digiIdx = _digiIdx;
}

int L1RpcLogHit::getDigiIdx() const {
  return digiIdx;
}
