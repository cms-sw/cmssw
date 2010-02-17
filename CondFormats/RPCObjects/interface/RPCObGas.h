#ifndef RPCObGas_h
#define RPCObGas_h
#include <vector>

class RPCObGas {
    public:
      struct Item {
        int dpid;
        float flowin;
        float flowout;
        int day;
        int time;
      };
    RPCObGas(){}
    virtual ~RPCObGas(){}
    std::vector<Item>  ObGas_rpc;
   };

#endif

