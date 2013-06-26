/*
 * Payload definition(s): Gas Map (RPCObGasmap)
 *
 *  $Date: 2009/11/16 13:01:46 $
 *  $Revision: 1.2 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#ifndef RPCObGasmap_h
#define RPCObGasmap_h
#include <vector>
#include <string>


class RPCObGasmap {
    public:
      struct GasMap_Item {
        int dpid;
        int region;
        int ring;
        int station;
        int sector;
        int layer;
        int subsector;
        int suptype;
    };
    RPCObGasmap(){}
    virtual ~RPCObGasmap(){}
    std::vector<GasMap_Item> ObGasMap_rpc;
   };

#endif

