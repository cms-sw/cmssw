#ifndef L1Trigger_RPCMuon_h
#define L1Trigger_RPCMuon_h

/** \class RPCMuon
 * Basic L1RPC muon candidate. Containes coordinates of LogCone, in which the
 * muon was found, ptCode (0 - 31, 0 means no muon), track quality (depends on
 * count of fired planes), sign and number of pattern, that was fit to hits by m_PAC
 * \author Karol Bunkowski (Warsaw)
 *
 */

#include "L1Trigger/RPCTrigger/interface/RPCConst.h"
#include <vector>
//---------------------------------------------------------------------------
class RPCMuon {
public:
  ///Default constructor. No muon.
  RPCMuon();

  ///Constructor. All parameters are set.
  RPCMuon(const RPCConst::l1RpcConeCrdnts coneCrdnts,
          int ptCode, int quality, int sign, int patternNum, unsigned short firedPlanes);
  
  ///Constructor.
  RPCMuon(int ptCode, int quality, int sign, int patternNum, unsigned short firedPlanes);

  RPCConst::l1RpcConeCrdnts getConeCrdnts() const;

  void setConeCrdnts(const RPCConst::l1RpcConeCrdnts& coneCrdnts);

  int getTower() const;

  int getLogSector() const;

  int getLogSegment() const;

  void setPtCode(int ptCode);

  int getPtCode() const;

  void setQuality(int quality);

  int getQuality() const;

  void setSign(int sign);

  int getSign() const;

  int getPatternNum() const;

  void setPatternNum(int patternNum);

  void setLogConeIdx(int logConeIdx);

  ///the index in LogConesVec stored in L1RpcTrigg (accessed by GetActiveCones)
  int getLogConeIdx() const;

  ///bits of this number denotes fired planes that conform to pattern pattern
  unsigned short getFiredPlanes() const;

  void setRefStripNum(int refStripNum);

/** continous number of strip in reference plane, set by	RPCPacData::run
  * int refStripNum =
  * getPattern(bestMuon.getPatternNum()).getStripFrom(m_REF_PLANE[abs(m_CurrConeCrdnts.m_Tower)])
  * + m_CurrConeCrdnts.m_LogSector * 96 + m_CurrConeCrdnts.m_LogSegment * 8;
  */
  int getRefStripNum() const;
 
 
  struct TDigiLink {
     TDigiLink(short int l, short int d) : m_layer(l), m_digiIdx(d) {};
     short int m_layer;
     short int m_digiIdx;  // vec?
  };

  typedef std::vector<TDigiLink > TDigiLinkVec; 

  TDigiLinkVec getDigiIdxVec() const {return m_digiIdxVec;};
  void setDigiIdxVec(const TDigiLinkVec& d) {m_digiIdxVec = d;};
  
protected:
  ///The coordinates of LogCone, in which the muon was found.
  RPCConst::l1RpcConeCrdnts m_ConeCrdnts;

  ///5 bits, 0-31.
  unsigned int m_PtCode;

  ///3 bits, 0-7.
  unsigned int m_Quality;

  ///1 bit, 0 - negative, 1 - positive.
  unsigned int m_Sign;

  ///number of pattern (in given Pac), that was fit to this muon.
  int m_PatternNum;  

  int m_LogConeIdx;

  ///bits of this number denotes fired planes that conform to pattern pattern 
  unsigned short m_FiredPlanes;

  int m_RefStripNum;


  TDigiLinkVec m_digiIdxVec;

};

typedef std::vector<RPCMuon> L1RpcMuonsVec;
#endif
