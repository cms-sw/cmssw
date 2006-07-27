//---------------------------------------------------------------------------
#ifndef L1RpcBasicTrigConfigH
#define L1RpcBasicTrigConfigH


#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/RPCTrigger/src/L1RpcTriggerConfiguration.h"
#include "L1Trigger/RPCTrigger/src/L1RpcPacManager.h"
#include "L1Trigger/RPCTrigger/src/L1RpcConst.h"
//#include "L1Trigger/RPCTrigger/src/L1RpcParametersDef.h"
#include "L1Trigger/RPCTrigger/src/L1RpcParameters.h"
//#include "L1Trigger/RPCTrigger/src/L1RpcException.h"
//---------------------------------------------------------------------------
class L1RpcBasicTrigConfig: public L1RpcTriggerConfiguration {
public:
  L1RpcBasicTrigConfig(L1RpcPacManager<L1RpcPac>* pacManager);
  
  L1RpcBasicTrigConfig();

  ///returns count of Trigger Crates in system.
  int GetTCsCnt();

  ///returns number og Trigger Boards in one Trigger Crate.
  int GetTBsInTC();

  /** One TB covers 3 or 4 Towers. The function returns the index of tower
    * on TB. */
  int GetTowerNumOnTb(const rpcparam::L1RpcConeCrdnts& coneCrdnts);

  ///Returns pointer to PAC that should run given LogCone. The PACs are holded by L1PacManager.
  const L1RpcPac* GetPac(const rpcparam::L1RpcConeCrdnts& coneCrdnts) const;

  ///Returns the index of TC that should run given LogCone.
  int GetTCNum(const rpcparam::L1RpcConeCrdnts& coneCrdnts);

  ///Returns the index of TB (in TC) that should run given LogCone.
  int GetTBNum(const rpcparam::L1RpcConeCrdnts& coneCrdnts);

  ///Returns the count of Towers (3 or 4), that are covered by given TB.
  int GetTowsCntOnTB(int tbNum);

  /** Converts TC GB-Sorter input tower address <0...35> ("tower number natural")
    * to tower number <-16...0...16>
    * TC GB-Sorter input tower address is 8 bits: [7...2] TB num, [1...0] tower num on TB.*/
  int TowAddr2TowNum(int towAddr);

  /** Converts TC GB-Sorter output tower address <0...31> ("tower number continous")
    * to tower number 2'complement*/
  int TowNum2TowNum2Comp(int towNum);

private:
  static const int TRIGGER_CRATES_CNT;

  static const int TB_IN_TC_CNT = 9;

  static const int TOWER_ON_TB[2 * L1RpcConst::ITOW_MAX + 1 +1];

  static const int TOWERS_CNT_ON_TB[TB_IN_TC_CNT];

  static const int TB_NUM_FOR_TOWER[2 * L1RpcConst::ITOW_MAX + 1];

  static const int TOW_ADDR_2_TOW_NUM[36];

  L1RpcPacManager<L1RpcPac>* PacManager;
};
#endif
