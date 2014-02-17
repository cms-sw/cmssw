/*
 * Payload definition(s): Feb Map (RPCObFebAssmap)
 *
 *  $Date: 2009/11/16 12:56:29 $
 *  $Revision: 1.2 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#ifndef RPCObFebAssmap_h
#define RPCObFebAssmap_h
#include <vector>
#include <string>


class RPCObFebAssmap {
    public:
      struct FebAssmap_Item {
        int dpid;
        int region;
        int ring;
        int station;
        int sector;
        int layer;
        int subsector;
        int chid;
    };
    RPCObFebAssmap(){}
    virtual ~RPCObFebAssmap(){}
    std::vector<FebAssmap_Item> ObFebAssmap_rpc;
   };

#endif

