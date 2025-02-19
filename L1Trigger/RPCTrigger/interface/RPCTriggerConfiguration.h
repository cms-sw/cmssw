#ifndef L1Trigger_RPCTriggerConfiguration_h
#define L1Trigger_RPCTriggerConfiguration_h

/** \class RPCTriggerConfiguration
  * Interface for classes storing the configuration of hardware in m_PAC trigger system.
  * \author Karol Bunkowski (Warsaw)
  */

#include "L1Trigger/RPCTrigger/interface/RPCPacData.h"

//interface class for storing the geometry of trigger
//f.e. which pac, or TB or TC should run given LogCone
class RPCTriggerConfiguration {
public:
  ///returns count of Trigger Crates in system.
  virtual int getTCsCnt() = 0;

  ///returns number og Trigger Boards in one Trigger Crate.
  virtual int getTBsInTC() = 0;

  /** Returns the index of m_tower on TB.*/
  virtual int getTowerNumOnTb(const RPCConst::l1RpcConeCrdnts& coneCrdnts) = 0;

  ///Returns pointer to m_PAC that should run given LogCone.
  virtual const RPCPacData* getPac(const RPCConst::l1RpcConeCrdnts& coneCrdnts) const = 0;

  ///Returns the index of TC that should run given LogCone.
  virtual int getTCNum(const RPCConst::l1RpcConeCrdnts& coneCrdnts) = 0;

  ///Returns the index of TB (in TC) that should run given LogCone.
  virtual int getTBNum(const RPCConst::l1RpcConeCrdnts& coneCrdnts) = 0;

  ///Returns the count of Towers, that are covered by given TB .
  virtual int getTowsCntOnTB(int tbNum) = 0;

  virtual int towAddr2TowNum(int towAddr) = 0;

  virtual int towNum2TowNum2Comp(int towNum) = 0;

  int getDebugLevel() const {
    return m_DebugLevel;
  }

  void setDebugLevel(int debgLevel) {
    m_DebugLevel = debgLevel;
  }

private:
  int m_DebugLevel;
};
#endif
