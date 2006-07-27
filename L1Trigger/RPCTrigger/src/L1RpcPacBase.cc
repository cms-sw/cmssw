/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2002                                                      *
*                                                                              *
*******************************************************************************/
#include "L1Trigger/RPCTrigger/src/L1RpcPacBase.h"
/**
 *
 * Constructor. ConeCrdnts and  CurrConeCrdnts are set.
 *
 */
L1RpcPacBase::L1RpcPacBase(int tower, int logSector, int logSegment) {
  ConeCrdnts.Tower = tower;
  ConeCrdnts.LogSector = logSector;
  ConeCrdnts.LogSegment = logSegment;

  CurrConeCrdnts = ConeCrdnts;
}

/**
 *
 *Constructor. ConeCrdnts and  CurrConeCrdnts are set.
 *
 */
L1RpcPacBase::L1RpcPacBase(L1RpcConst::L1RpcConeCrdnts coneCrdnts): ConeCrdnts(coneCrdnts), CurrConeCrdnts(coneCrdnts) {}

/**
 *
 * CurrConeCrdnts are set. Called by L1RpcPacManager in GetPac.
 *
 */
void L1RpcPacBase::SetCurrentPosition(int tower, int logSector, int logSegment) {
  CurrConeCrdnts.Tower = tower;
  CurrConeCrdnts.LogSector = logSector;
  CurrConeCrdnts.LogSegment = logSegment;
}

/**
 *
 *CurrConeCrdnts are set. Called by L1RpcPacManager in GetPac.
 *
 */
void L1RpcPacBase::SetCurrentPosition(L1RpcConst::L1RpcConeCrdnts coneCrdnts) {
  CurrConeCrdnts = coneCrdnts;
}
