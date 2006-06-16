//---------------------------------------------------------------------------

#ifndef L1RpcMuonH
#define L1RpcMuonH

/** \class L1RpcMuon
 * Basic L1RPC muon candidate. Containes coordinates of LogCone, in which the
 * muon was found, ptCode (0 - 31, 0 means no muon), track quality (depends on
 * count of fired planes), sign and number of pattern, that was fit to hits by PAC
 * \author Karol Bunkowski (Warsaw)
 *
 */

//#include "L1Trigger/RPCTrigger/src/L1RpcParametersDef.h"
#include "L1Trigger/RPCTrigger/src/L1RpcParameters.h"
#include <vector>
//---------------------------------------------------------------------------
class L1RpcMuon {
public:
  ///Default constructor. No muon.
  L1RpcMuon();

  ///Constructor. All parameters are set.
  L1RpcMuon(const RPCParam::L1RpcConeCrdnts coneCrdnts, int ptCode, int quality, int sign, int patternNum, unsigned short firedPlanes);
  
  ///Constructor.
  L1RpcMuon(int ptCode, int quality, int sign, int patternNum, unsigned short firedPlanes);

  RPCParam::L1RpcConeCrdnts GetConeCrdnts() const;

  void SetConeCrdnts(const RPCParam::L1RpcConeCrdnts& coneCrdnts);

  int GetTower() const;

  int GetLogSector() const;

  int GetLogSegment() const;

  void SetPtCode(int ptCode);

  int GetPtCode() const;

  void SetQuality(int quality);

  int GetQuality() const;

  void SetSign(int sign);

  int GetSign() const;

  int GetPatternNum() const;

  void SetPatternNum(int patternNum);

  void SetLogConeIdx(int logConeIdx);

  ///the index in LogConesVec stored in L1RpcTrigg (accessed by GetActiveCones)
  int GetLogConeIdx() const;

  ///bits of this number denotes fired planes that conform to pattern pattern
  unsigned short GetFiredPlanes() const;

  void SetRefStripNum(int refStripNum);

  /** continous number of strip in reference plane, set by	L1RpcPac::Run
    * int refStripNum = GetPattern(bestMuon.GetPatternNum()).GetStripFrom(REF_PLANE[abs(CurrConeCrdnts.Tower)]) + CurrConeCrdnts.LogSector * 96 + CurrConeCrdnts.LogSegment * 8; 
    */
  int GetRefStripNum() const;
    
protected:
  ///The coordinates of LogCone, in which the muon was found.
  RPCParam::L1RpcConeCrdnts ConeCrdnts;

  ///5 bits, 0-31.
  unsigned int PtCode;

  ///3 bits, 0-7.
  unsigned int Quality;

  ///1 bit, 0 - negative, 1 - positive.
  unsigned int Sign;

  ///number of pattern (in given Pac), that was fit to this muon.
  int PatternNum;  

  int LogConeIdx;

  ///bits of this number denotes fired planes that conform to pattern pattern 
  unsigned short FiredPlanes;

  int RefStripNum;
};

typedef std::vector<L1RpcMuon> L1RpcMuonsVec;
#endif
