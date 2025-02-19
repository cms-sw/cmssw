/*
 * Payload definition(s): DpId Map for RPCObCond Payload 
 *
 *  $Date: 2009/11/16 13:03:27 $
 *  $Revision: 1.3 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#ifndef RPCObPVSSmap_h
#define RPCObPVSSmap_h
#include <vector>
#include <string>


class RPCObPVSSmap {
    public:
      struct Item {
        int since;
        int dpid;
        int region;
        int ring;
        int station;
        int sector;
        int layer;
        int subsector;
        int suptype;
    };
    RPCObPVSSmap(){}
    virtual ~RPCObPVSSmap(){}
    std::vector<Item> ObIDMap_rpc;
   };

#endif

