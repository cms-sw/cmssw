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

