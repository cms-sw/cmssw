/*
 * Payload definition(s): UXC Temperature, Pressure and Humidity (RPCObUXC)
 *
 *  $Date: 2009/11/16 13:04:37 $
 *  $Revision: 1.3 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#ifndef RPCObUXC_h
#define RPCObUXC_h
#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>

class RPCObUXC {
    public:
      struct Item {
        float temperature;
        float pressure;
        float dewpoint;
	int unixtime;
      
  COND_SERIALIZABLE;
};
    RPCObUXC(){}
    virtual ~RPCObUXC(){}
    std::vector<Item>  ObUXC_rpc;
   
  COND_SERIALIZABLE;
};

#endif

