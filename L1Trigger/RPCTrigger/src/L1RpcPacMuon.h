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
  
  L1RpcPacMuon();
  
  L1RpcPacMuon(const L1RpcPattern& pattern, int quality, unsigned short firedPlanes);

  void SetAll(const L1RpcPattern& pattern, int quality, unsigned short firedPlanes);

  void SetPatternNum(int patternNum);

  bool operator < (const L1RpcPacMuon& pacMuon) const;

  bool operator > (const L1RpcPacMuon& pacMuon) const;

  bool operator == (const L1RpcPacMuon& pacMuon) const;

private:
};
#endif


