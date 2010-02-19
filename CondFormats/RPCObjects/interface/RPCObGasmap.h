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

