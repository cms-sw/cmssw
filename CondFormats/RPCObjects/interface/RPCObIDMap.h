#ifndef RPCObIDMap_h
#define RPCObIDMap_h
#include <vector>
#include <string>


class RPCObIDMap {
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
    RPCObIDMap(){}
    virtual ~RPCObIDMap(){}
    std::vector<Item> ObIDMap_rpc;
   };

#endif

