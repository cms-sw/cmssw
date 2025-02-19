/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2002                                                      *
*                                                                              *
*******************************************************************************/
#include "L1Trigger/RPCTrigger/interface/RPCPacBase.h"
/**
 *
 * Constructor. m_ConeCrdnts and  m_CurrConeCrdnts are set.
 *
 */
RPCPacBase::RPCPacBase(int m_tower, int logSector, int logSegment) {
  m_ConeCrdnts.m_Tower = m_tower;
  m_ConeCrdnts.m_LogSector = logSector;
  m_ConeCrdnts.m_LogSegment = logSegment;

  m_CurrConeCrdnts = m_ConeCrdnts;
}

/**
 *
 *Constructor. m_ConeCrdnts and  m_CurrConeCrdnts are set.
 *
 */
RPCPacBase::RPCPacBase(RPCConst::l1RpcConeCrdnts coneCrdnts): 
    m_ConeCrdnts(coneCrdnts), 
    m_CurrConeCrdnts(coneCrdnts) {}

/**
 *
 * m_CurrConeCrdnts are set. Called by RPCPacManager in getPac.
 *
 */
void RPCPacBase::setCurrentPosition(int m_tower, int logSector, int logSegment) {
  m_CurrConeCrdnts.m_Tower = m_tower;
  m_CurrConeCrdnts.m_LogSector = logSector;
  m_CurrConeCrdnts.m_LogSegment = logSegment;
}

/**
 *
 *m_CurrConeCrdnts are set. Called by RPCPacManager in getPac.
 *
 */
void RPCPacBase::setCurrentPosition(RPCConst::l1RpcConeCrdnts coneCrdnts) {
  m_CurrConeCrdnts = coneCrdnts;
}
