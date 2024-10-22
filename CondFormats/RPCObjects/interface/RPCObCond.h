/*
 * Payload definition(s): Current (RPCObImon), High Voltage (RPCObVmon), Chamber Status (RPCObStatus) 
 *
 *  $Date: 2009/11/10 12:20:23 $
 *  $Revision: 1.17 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#ifndef RPCObCond_h
#define RPCObCond_h
#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>

class RPCObImon {
public:
  struct I_Item {
    int dpid;
    float value;
    int day;
    int time;

    COND_SERIALIZABLE;
  };
  RPCObImon() {}
  virtual ~RPCObImon() {}
  std::vector<I_Item> ObImon_rpc;

  COND_SERIALIZABLE;
};

class RPCObVmon {
public:
  struct V_Item {
    int dpid;
    float value;
    int day;
    int time;

    COND_SERIALIZABLE;
  };
  RPCObVmon() {}
  virtual ~RPCObVmon() {}
  std::vector<V_Item> ObVmon_rpc;

  COND_SERIALIZABLE;
};

class RPCObStatus {
public:
  struct S_Item {
    int dpid;
    float value;
    int day;
    int time;

    COND_SERIALIZABLE;
  };
  RPCObStatus() {}
  virtual ~RPCObStatus() {}
  std::vector<S_Item> ObStatus_rpc;

  COND_SERIALIZABLE;
};

class RPCObTemp {
public:
  struct T_Item {
    int dpid;
    float value;
    int day;
    int time;

    COND_SERIALIZABLE;
  };
  RPCObTemp() {}
  virtual ~RPCObTemp() {}
  std::vector<T_Item> ObTemp_rpc;

  COND_SERIALIZABLE;
};

#endif
