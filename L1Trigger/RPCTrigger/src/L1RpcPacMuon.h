/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2002                                                      *
*                                                                              *
*******************************************************************************/

#ifndef L1RpcPacMuonH
#define L1RpcPacMuonH
/** \class L1RpcPacMuon
 *
 * Muon candidate found by PAC for one LogCone. Containes the compare operators
 * used during sorting inside the PAC. The PAC sets for muon its cone coordinates.
 * \author Karol Bunkowski (Warsaw)
 */

//#include "L1Trigger/RPCTrigger/src/L1RpcParametersDef.h"
#include "L1Trigger/RPCTrigger/src/L1RpcParameters.h"
#include "L1Trigger/RPCTrigger/src/L1RpcMuon.h"
#include "L1Trigger/RPCTrigger/src/L1RpcPattern.h"
//------------------------------------------------------------------------------

class L1RpcPacMuon: public L1RpcMuon {
public:
  ///Default constructor. No muon.
  L1RpcPacMuon(): L1RpcMuon() {
  }

  //Constructor.
  L1RpcPacMuon(const L1RpcPattern& pattern, int quality, unsigned short firedPlanes):
    L1RpcMuon(pattern.GetCode(), quality, pattern.GetSign(), pattern.GetNumber(), firedPlanes) {
  }

  void SetAll(const L1RpcPattern& pattern, int quality, unsigned short firedPlanes) {
    PatternNum = pattern.GetNumber();
    PtCode = pattern.GetCode();
    Sign = pattern.GetSign();
    Quality = quality;
    FiredPlanes = firedPlanes;
  }

  void SetPatternNum(int patternNum) {
    PatternNum = patternNum;
  };

  bool operator < (const L1RpcPacMuon& pacMuon) const;

  bool operator > (const L1RpcPacMuon& pacMuon) const;

  bool operator == (const L1RpcPacMuon& pacMuon) const;

private:
};
#endif


