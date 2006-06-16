#include "L1Trigger/RPCTrigger/src/L1RpcMuon.h"
//---------------------------------------------------------------------------

//#############################################################################################
//
///Default constructor. No muon.
//
//#############################################################################################
L1RpcMuon::L1RpcMuon() {
  PtCode = 0;
  Quality = 0;
  Sign = 0;

  PatternNum = -1;
  RefStripNum = -1;
}
//#############################################################################################
//
///Constructor. All parameters are set.
//
//#############################################################################################
L1RpcMuon::L1RpcMuon(const RPCParam::L1RpcConeCrdnts coneCrdnts, int ptCode, int quality, int sign, 
          int patternNum, unsigned short firedPlanes) 
{
    ConeCrdnts = coneCrdnts;
    PtCode = ptCode;
    Quality = quality;
    Sign = sign;
    PatternNum = patternNum;
    FiredPlanes = firedPlanes;
}
//#############################################################################################
//
///Constructor.
//
//#############################################################################################
L1RpcMuon::L1RpcMuon(int ptCode, int quality, int sign, int patternNum, unsigned short firedPlanes) {
  
  ConeCrdnts = RPCParam::L1RpcConeCrdnts();
  PtCode = ptCode;
  Quality = quality;
  Sign = sign;
  PatternNum = patternNum;
  FiredPlanes = firedPlanes;
}
//#############################################################################################
//
//  Simple getters and setters
//
//#############################################################################################
RPCParam::L1RpcConeCrdnts L1RpcMuon::GetConeCrdnts() const {  return ConeCrdnts; }

void L1RpcMuon::SetConeCrdnts(const RPCParam::L1RpcConeCrdnts& coneCrdnts) { ConeCrdnts = coneCrdnts; }

int L1RpcMuon::GetTower() const { return ConeCrdnts.Tower;}

int L1RpcMuon::GetLogSector() const { return ConeCrdnts.LogSector; }

int L1RpcMuon::GetLogSegment() const { return ConeCrdnts.LogSegment; }

void L1RpcMuon::SetPtCode(int ptCode) { PtCode = ptCode; }

int L1RpcMuon::GetPtCode() const { return PtCode; }

void L1RpcMuon::SetQuality(int quality) { Quality = quality; }

int L1RpcMuon::GetQuality() const { return Quality; }

void L1RpcMuon::SetSign(int sign) { Sign = sign; }

int L1RpcMuon::GetSign() const { return Sign; }

int L1RpcMuon::GetPatternNum() const { return PatternNum; }

void L1RpcMuon::SetPatternNum(int patternNum) { PatternNum = patternNum; }

void L1RpcMuon::SetLogConeIdx(int logConeIdx) { LogConeIdx = logConeIdx; }

///the index in LogConesVec stored in L1RpcTrigg (accessed by GetActiveCones)
int L1RpcMuon::GetLogConeIdx() const { return LogConeIdx; }

///bits of this number denote fired planes that conform to pattern pattern
unsigned short L1RpcMuon::GetFiredPlanes() const { return FiredPlanes; }

void L1RpcMuon::SetRefStripNum(int refStripNum) { RefStripNum = refStripNum; }

/** continous number of strip in reference plane, set by	L1RpcPac::Run
  * nt refStripNum = GetPattern(bestMuon.GetPatternNum()).GetStripFrom(REF_PLANE[abs(CurrConeCrdnts.Tower)]) +
  *    CurrConeCrdnts.LogSector * 96 + CurrConeCrdnts.LogSegment * 8; 
  */
int L1RpcMuon::GetRefStripNum() const { return RefStripNum; }
