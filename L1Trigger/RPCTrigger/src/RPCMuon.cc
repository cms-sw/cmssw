#include "L1Trigger/RPCTrigger/interface/RPCMuon.h"
//---------------------------------------------------------------------------

//#############################################################################################
//
///Default constructor. No muon.
//
//#############################################################################################
RPCMuon::RPCMuon() {
  m_PtCode = 0;
  m_Quality = 0;
  m_Sign = 0;

  m_PatternNum = -1;
  m_RefStripNum = -1;
}
//#############################################################################################
//
///Constructor. All parameters are set.
//
//#############################################################################################
RPCMuon::RPCMuon(const RPCConst::l1RpcConeCrdnts coneCrdnts,
                 int ptCode, int quality, int sign,
                 int patternNum, unsigned short firedPlanes)
{
    m_ConeCrdnts = coneCrdnts;
    m_PtCode = ptCode;
    m_Quality = quality;
    m_Sign = sign;
    m_PatternNum = patternNum;
    m_FiredPlanes = firedPlanes;
}
//#############################################################################################
//
///Constructor.
//
//#############################################################################################
RPCMuon::RPCMuon(int ptCode, int quality, int sign, int patternNum, unsigned short firedPlanes) {
  
  m_ConeCrdnts = RPCConst::l1RpcConeCrdnts();
  m_PtCode = ptCode;
  m_Quality = quality;
  m_Sign = sign;
  m_PatternNum = patternNum;
  m_FiredPlanes = firedPlanes;
}
//#############################################################################################
//
//  Simple getters and setters
//
//#############################################################################################
RPCConst::l1RpcConeCrdnts RPCMuon::getConeCrdnts() const {  return m_ConeCrdnts; }

void RPCMuon::setConeCrdnts(const RPCConst::l1RpcConeCrdnts& coneCrdnts) {
  m_ConeCrdnts = coneCrdnts;
}

int RPCMuon::getTower() const { return m_ConeCrdnts.m_Tower;}

int RPCMuon::getLogSector() const { return m_ConeCrdnts.m_LogSector; }

int RPCMuon::getLogSegment() const { return m_ConeCrdnts.m_LogSegment; }

void RPCMuon::setPtCode(int ptCode) { m_PtCode = ptCode; }

int RPCMuon::getPtCode() const { return m_PtCode; }

void RPCMuon::setQuality(int quality) { m_Quality = quality; }

int RPCMuon::getQuality() const { return m_Quality; }

void RPCMuon::setSign(int sign) { m_Sign = sign; }

int RPCMuon::getSign() const { return m_Sign; }

int RPCMuon::getPatternNum() const { return m_PatternNum; }

void RPCMuon::setPatternNum(int patternNum) { m_PatternNum = patternNum; }

void RPCMuon::setLogConeIdx(int logConeIdx) { m_LogConeIdx = logConeIdx; }

///the index in LogConesVec stored in L1RpcTrigg (accessed by GetActiveCones)
int RPCMuon::getLogConeIdx() const { return m_LogConeIdx; }

///bits of this number denote fired planes that conform to pattern pattern
unsigned short RPCMuon::getFiredPlanes() const { return m_FiredPlanes; }

void RPCMuon::setRefStripNum(int refStripNum) { m_RefStripNum = refStripNum; }

/** continous number of strip in reference plane, set by	RPCPacData::run
  * int refStripNum = 
  *   getPattern(bestMuon.getPatternNum()).getStripFrom(m_REF_PLANE[abs(m_CurrConeCrdnts.m_Tower)])
  *   + m_CurrConeCrdnts.m_LogSector * 96 + m_CurrConeCrdnts.m_LogSegment * 8; 
  */
int RPCMuon::getRefStripNum() const { return m_RefStripNum; }
