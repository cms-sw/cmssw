#ifndef RPCObUXC_h
#define RPCObUXC_h
#include <vector>

class RPCObUXC {
    public:
      struct Item {
        float temperature;
        float pressure;
	int unixtime;
      };
    RPCObUXC(){}
    virtual ~RPCObUXC(){}
    std::vector<Item>  ObUXC_rpc;
   };

#endif

