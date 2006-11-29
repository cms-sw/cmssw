//---------------------------------------------------------------------------
#ifndef L1RpcBasicTrigConfigH
#define L1RpcBasicTrigConfigH

#ifndef _STAND_ALONE
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#endif //_STAND_ALONE

#include "L1Trigger/RPCTrigger/src/RPCTriggerConfiguration.h"
#include "L1Trigger/RPCTrigger/src/RPCPacManager.h"
#include "L1Trigger/RPCTrigger/src/RPCConst.h"
#include "L1Trigger/RPCTrigger/src/RPCConst.h"
//#include "L1Trigger/RPCTrigger/src/RPCException.h"
//---------------------------------------------------------------------------
class RPCBasicTrigConfig: public RPCTriggerConfiguration {
public:
  RPCBasicTrigConfig(RPCPacManager<RPCPacData>* pacManager);
  
  RPCBasicTrigConfig();

  ///returns count of Trigger Crates in system.
  int getTCsCnt();

  ///returns number og Trigger Boards in one Trigger Crate.
  int getTBsInTC();

  /** One TB covers 3 or 4 Towers. The function returns the index of m_tower
    * on TB. */
  int getTowerNumOnTb(const RPCConst::l1RpcConeCrdnts& coneCrdnts);

  ///Returns pointer to m_PAC that should run given LogCone. The PACs are holded by L1PacManager.
  const RPCPacData* getPac(const RPCConst::l1RpcConeCrdnts& coneCrdnts) const;

  ///Returns the index of TC that should run given LogCone.
  int getTCNum(const RPCConst::l1RpcConeCrdnts& coneCrdnts);

  ///Returns the index of TB (in TC) that should run given LogCone.
  int getTBNum(const RPCConst::l1RpcConeCrdnts& coneCrdnts);

  ///Returns the count of Towers (3 or 4), that are covered by given TB.
  int getTowsCntOnTB(int tbNum);

  /** Converts TC GB-Sorter input m_tower address <0...35> ("m_tower number natural")
    * to m_tower number <-16...0...16>
    * TC GB-Sorter input m_tower address is 8 bits: [7...2] TB num, [1...0] m_tower num on TB.*/
  int towAddr2TowNum(int towAddr);

  /** Converts TC GB-Sorter output m_tower address <0...31> ("m_tower number continous")
    * to m_tower number 2'complement*/
  int towNum2TowNum2Comp(int towNum);

private:
  static const int m_TRIGGER_CRATES_CNT;

  static const int m_TB_IN_TC_CNT = 9;

  static const int m_TOWER_ON_TB[2 * RPCConst::ITOW_MAX + 1 +1];

  static const int m_TOWERS_CNT_ON_TB[m_TB_IN_TC_CNT];

  static const int m_TB_NUM_FOR_TOWER[2 * RPCConst::ITOW_MAX + 1];

  static const int m_TOW_ADDR_2_TOW_NUM[36];

  RPCPacManager<RPCPacData>* m_PacManager;
};
#endif
