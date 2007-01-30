#include "L1Trigger/RPCTrigger/interface/RPCLogHit.h"


/**
 *
 *\brief Default ctor
 *\todo check m_LogSector/m_LogSegment convention
 *
*/
RPCLogHit::RPCLogHit(int m_tower, int m_PAC, int m_logplane, int m_posInCone){

  m_ConeCrdnts.m_Tower=m_tower;
  m_ConeCrdnts.m_LogSector=m_PAC/12;
  m_ConeCrdnts.m_LogSegment=m_PAC%12;
  m_stripNumberInCone = m_posInCone;
  m_logPlaneNumber = m_logplane;
  
}
//###################################################################
//
// Simple getters and setters
//
//###################################################################
RPCConst::l1RpcConeCrdnts RPCLogHit::getConeCrdnts() const {
  return m_ConeCrdnts;
}

int RPCLogHit::getTower() const {
  return m_ConeCrdnts.m_Tower;
}

int RPCLogHit::getLogSector() const {
  return m_ConeCrdnts.m_LogSector;
}

int RPCLogHit::getLogSegment() const {
  return m_ConeCrdnts.m_LogSegment;
}

int RPCLogHit::getlogPlaneNumber() const {
  return m_logPlaneNumber;
}

int RPCLogHit::getStripNumberInCone() const {
  return m_stripNumberInCone;
}

void  RPCLogHit::setDigiIdx(int digiIdx) {
  m_digiIdx = digiIdx;
}

int RPCLogHit::getDigiIdx() const {
  return m_digiIdx;
}
