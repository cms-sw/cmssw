//---------------------------------------------------------------------------

#ifndef L1RpcTriggerConfigurationH
#define L1RpcTriggerConfigurationH

/** \class L1RpcTriggerConfiguration
  * Interface for classes storing the configuration of hardware in PAC trigger system.
  * \author Karol Bunkowski (Warsaw)
  */

#include "L1Trigger/RPCTrigger/src/L1RpcPac.h"

//interface class for storing the geometry of trigger
//f.e. which pac, or TB or TC should run given LogCone
class L1RpcTriggerConfiguration {
public:
  ///returns count of Trigger Crates in system.
  virtual int GetTCsCnt() = 0;

  ///returns number og Trigger Boards in one Trigger Crate.
  virtual int GetTBsInTC() = 0;

  /** Returns the index of tower on TB.*/
  virtual int GetTowerNumOnTb(const rpcparam::L1RpcConeCrdnts& coneCrdnts) = 0;

  ///Returns pointer to PAC that should run given LogCone.
  virtual const L1RpcPac* GetPac(const rpcparam::L1RpcConeCrdnts& coneCrdnts) const = 0;

  ///Returns the index of TC that should run given LogCone.
  virtual int GetTCNum(const rpcparam::L1RpcConeCrdnts& coneCrdnts) = 0;

  ///Returns the index of TB (in TC) that should run given LogCone.
  virtual int GetTBNum(const rpcparam::L1RpcConeCrdnts& coneCrdnts) = 0;

  ///Returns the count of Towers, that are covered by given TB .
  virtual int GetTowsCntOnTB(int tbNum) = 0;

  virtual int TowAddr2TowNum(int towAddr) = 0;

  virtual int TowNum2TowNum2Comp(int towNum) = 0;

  int GetDebugLevel() {
    return DebugLevel;
  }

  void SetDebugLevel(int debgLevel) {
    DebugLevel = debgLevel;
  }

private:
  int DebugLevel;
};
#endif
