#ifndef RPCObCond_h
#define RPCObCond_h
#include <vector>

class RPCObCond {
    public:
      struct Item {
        int dpid;
        float value;
        int day;
        int time;
      };
    RPCObCond(){}
    virtual ~RPCObCond(){}
    std::vector<Item> ObImon_rpc;
    std::vector<Item> ObVmon_rpc;
    std::vector<Item> ObStatus_rpc;
    std::vector<Item> ObTemp_rpc;
   };

#endif

