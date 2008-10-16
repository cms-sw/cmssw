#ifndef RPCObPVSSmap_h
#define RPCObPVSSmap_h
#include <vector>
#include <string>


class RPCObPVSSmap {
    public:
      struct Item {
        std::string since;
        int dpid;
        std::string region;
        std::string ring;
        std::string station;
        std::string sector;
        std::string layer;
        std::string subsector;
        std::string suptype;
    };
    RPCObPVSSmap(){}
    virtual ~RPCObPVSSmap(){}
    std::vector<Item> ObIDMap_rpc;
   };

#endif

