#ifndef RPCObAlignment_h
#define RPCObAlignment_h
#include <vector>

class RPCObAlignment {
    public:
      struct Alignment_Item {

        int   dpid;
        float alocalX;
	float alocalPhi;
    };
    RPCObAlignment(){}
    virtual ~RPCObAlignment(){}
    std::vector<Alignment_Item> ObAlignment_rpc;
   };

#endif

