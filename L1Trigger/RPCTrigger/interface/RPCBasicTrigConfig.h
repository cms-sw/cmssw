#ifndef L1Trigger_RPCBasicTrigConfig_h
#define L1Trigger_RPCBasicTrigConfig_h

#ifndef _STAND_ALONE
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#endif //_STAND_ALONE

#include "L1Trigger/RPCTrigger/interface/RPCTriggerConfiguration.h"
#include "L1Trigger/RPCTrigger/interface/RPCPacManager.h"
#include "L1Trigger/RPCTrigger/interface/RPCConst.h"
#include "L1Trigger/RPCTrigger/interface/RPCConst.h"
//#include "L1Trigger/RPCTrigger/interface/RPCException.h"
//---------------------------------------------------------------------------
class RPCBasicTrigConfig: public RPCTriggerConfiguration {
public:
  RPCBasicTrigConfig(RPCPacManager<RPCPacData>* pacManager);
  
  RPCBasicTrigConfig();

  ///returns count of Trigger Crates in system.
  virtual int getTCsCnt();

  ///returns number og Trigger Boards in one Trigger Crate.
  virtual int getTBsInTC();

  /** One TB covers 3 or 4 Towers. The function returns the index of m_tower
    * on TB. */
  virtual int getTowerNumOnTb(const RPCConst::l1RpcConeCrdnts& coneCrdnts);

  ///Returns pointer to m_PAC that should run given LogCone. The PACs are holded by L1PacManager.
  virtual const RPCPacData* getPac(const RPCConst::l1RpcConeCrdnts& coneCrdnts) const;

  ///Returns the index of TC that should run given LogCone.
  virtual int getTCNum(const RPCConst::l1RpcConeCrdnts& coneCrdnts);

  ///Returns the index of TB (in TC) that should run given LogCone.
  virtual int getTBNum(const RPCConst::l1RpcConeCrdnts& coneCrdnts);

  ///Returns the count of Towers (3 or 4), that are covered by given TB.
  virtual int getTowsCntOnTB(int tbNum);

  /** Converts TC GB-Sorter input m_tower address <0...35> ("m_tower number natural")
    * to m_tower number <-16...0...16>
    * TC GB-Sorter input m_tower address is 8 bits: [7...2] TB num, [1...0] m_tower num on TB.*/
  virtual int towAddr2TowNum(int towAddr);

  /** Converts TC GB-Sorter output m_tower address <0...31> ("m_tower number continous")
    * to m_tower number 2'complement*/
  virtual int towNum2TowNum2Comp(int towNum);

  virtual ~RPCBasicTrigConfig() {}

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
