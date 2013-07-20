/*
 * Payload definition(s): Chamber Alignment (RPCObAlignment)
 *
 *  $Date: 2009/11/16 12:55:34 $
 *  $Revision: 1.4 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

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

