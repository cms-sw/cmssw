/*
 * Payload definition(s): Chamber Gas Humidity (RPCObGasHum)
 *
 *  $Date: 2009/12/14 16:00:11 $
 *  $Revision: 1.2 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#ifndef RPCObGasHum_h
#define RPCObGasHum_h
#include <vector>

class RPCObGasHum {
    public:
      struct Item {
        int unixtime;
        float value;
        int dpid;
      };
    RPCObGasHum(){}
    virtual ~RPCObGasHum(){}
    std::vector<Item>  ObGasHum_rpc;
   };

#endif

