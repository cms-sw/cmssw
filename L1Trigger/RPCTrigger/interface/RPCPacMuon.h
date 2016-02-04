#ifndef L1Trigger_RPCPacMuon_h
#define L1Trigger_RPCPacMuon_h
/** \class RPCPacMuon
 *
 * Muon candidate found by m_PAC for one LogCone. Containes the compare operators
 * used during sorting inside the m_PAC. The m_PAC sets for muon its cone coordinates.
 * \author Karol Bunkowski (Warsaw)
 */

#include "L1Trigger/RPCTrigger/interface/RPCConst.h"
#include "L1Trigger/RPCTrigger/interface/RPCMuon.h"
#include "CondFormats/L1TObjects/interface/RPCPattern.h"
//------------------------------------------------------------------------------

class RPCPacMuon: public RPCMuon {
public:
  
  RPCPacMuon();
  
  RPCPacMuon(const RPCPattern& pattern, int quality, unsigned short firedPlanes);

  void setAll(const RPCPattern& pattern, int quality, unsigned short firedPlanes);

  void setPatternNum(int patternNum);

  bool operator < (const RPCPacMuon& pacMuon) const;

  bool operator > (const RPCPacMuon& pacMuon) const;

  bool operator == (const RPCPacMuon& pacMuon) const;

private:
};
#endif


