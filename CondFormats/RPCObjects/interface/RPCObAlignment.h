#ifndef RPCObAlignment_h
#define RPCObAlignment_h
#include <vector>

class RPCObAlignment {
    public:
      struct Alignment_Item {

        int   dpid;
        float align;
    };
    RPCObAlignment(){}
    virtual ~RPCObAlignment(){}
    std::vector<Alignment_Item> ObAlignment_rpc;
   };

#endif

