/*
 * Payload definition(s): Current (RPCObImon), High Voltage (RPCObVmon), Chamber Status (RPCObStatus) 
 *
 *  $Date: 2009/11/16 12:53:47 $
 *  $Revision: 1.3 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#ifndef RPCObCond_h
#define RPCObCond_h
#include <vector>

class RPCObImon {
 public:
  struct I_Item {
    int detid;
    float value;
    unsigned int unixtime ;
  };
  RPCObImon(){}
  virtual ~RPCObImon(){}
  std::vector<I_Item> ObImon_rpc;
};

class RPCObVmon {
 public:
  struct V_Item {
    int detid;
    float value;
    unsigned int unixtime ;
  };
  RPCObVmon(){}
  virtual ~RPCObVmon(){}
  std::vector<V_Item> ObVmon_rpc;
};

class RPCObStatus {
 public:
  struct S_Item {
    int detid;
    float value;
    unsigned int unixtime ;
  };
  RPCObStatus(){}
  virtual ~RPCObStatus(){}
  std::vector<S_Item> ObStatus_rpc;
};

class RPCObTemp {
 public:
  struct T_Item {
    int detid;
    float value;
    unsigned int unixtime ;
  };
  RPCObTemp(){}
  virtual ~RPCObTemp(){}
  std::vector<T_Item> ObTemp_rpc;
};

#endif

