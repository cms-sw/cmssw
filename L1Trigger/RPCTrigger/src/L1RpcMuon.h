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
  L1RpcMuon() {
    PtCode = 0;
    Quality = 0;
    Sign = 0;

    PatternNum = -1;
    RefStripNum = -1;
  };

  ///Constructor. All parameters are set.
  L1RpcMuon(const RPCParam::L1RpcConeCrdnts coneCrdnts, int ptCode, int quality, int sign, int patternNum, unsigned short firedPlanes) {
    ConeCrdnts = coneCrdnts;

    PtCode = ptCode;
    Quality = quality;
    Sign = sign;

    PatternNum = patternNum;

    FiredPlanes = firedPlanes;
  };
  
  ///Constructor.
  L1RpcMuon(int ptCode, int quality, int sign, int patternNum, unsigned short firedPlanes) {
    ConeCrdnts = RPCParam::L1RpcConeCrdnts();

    PtCode = ptCode;
    Quality = quality;
    Sign = sign;

    PatternNum = patternNum;

    FiredPlanes = firedPlanes;
  };

  RPCParam::L1RpcConeCrdnts GetConeCrdnts() const {
    return ConeCrdnts;
  }

  void SetConeCrdnts(const RPCParam::L1RpcConeCrdnts& coneCrdnts) {
    ConeCrdnts = coneCrdnts;
  }

  int GetTower() const {
    return ConeCrdnts.Tower;
  }

  int GetLogSector() const {
    return ConeCrdnts.LogSector;
  }

  int GetLogSegment() const {
    return ConeCrdnts.LogSegment;
  }

  void SetPtCode(int ptCode) {
    PtCode = ptCode;
  };

  int GetPtCode() const {
    return PtCode;
  };

  void SetQuality(int quality) {
    Quality = quality;
  };

  int GetQuality() const {
    return Quality;
  };

  void SetSign(int sign) {
    Sign = sign;
  };

  int GetSign() const {
    return Sign;
  };

  int GetPatternNum() const {
    return PatternNum;
  }

  void SetPatternNum(int patternNum) {
    PatternNum = patternNum;
  }

  void SetLogConeIdx(int logConeIdx) {
    LogConeIdx = logConeIdx;
  }

  ///the index in LogConesVec stored in L1RpcTrigg (accessed by GetActiveCones)
  int GetLogConeIdx() const {
    return LogConeIdx;
  }

  ///bits of this number denotes fired planes that conform to pattern pattern
  unsigned short GetFiredPlanes() const {
    return FiredPlanes;
  }

	void SetRefStripNum(int refStripNum) {
		RefStripNum = refStripNum;
	}

  /** continous number of strip in reference plane, set by	L1RpcPac::Run
    * int refStripNum = GetPattern(bestMuon.GetPatternNum()).GetStripFrom(REF_PLANE[abs(CurrConeCrdnts.Tower)]) + CurrConeCrdnts.LogSector * 96 + CurrConeCrdnts.LogSegment * 8; 
    */
	int GetRefStripNum() const {
		return RefStripNum;
	}
  std::string ToString() const;
  
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
