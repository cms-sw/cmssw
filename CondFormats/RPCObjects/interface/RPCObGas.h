/*
 * Payload definition(s): Chamber Gas Flow In and Out (RPCObGas)
 *
 *  $Date: 2009/11/16 12:53:47 $
 *  $Revision: 1.3 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#ifndef RPCObGas_h
#define RPCObGas_h
#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>

class RPCObGas {
public:
  struct Item {
    int dpid;
    float flowin;
    float flowout;
    int day;
    int time;

    COND_SERIALIZABLE;
  };
  RPCObGas() {}
  virtual ~RPCObGas() {}
  std::vector<Item> ObGas_rpc;

  COND_SERIALIZABLE;
};

#endif
