#ifndef RPCRunIOV_h
#define RPCRunIOV_h
#include <vector>

class RPCRunIOV {
    public:
      struct RunIOV_Item {

        int run;
        int iov1;
        int iov2;
    };
    RPCRunIOV(){}
    virtual ~RPCRunIOV(){}
    std::vector<RunIOV_Item> ObRunIOV_rpc;
   };

#endif

