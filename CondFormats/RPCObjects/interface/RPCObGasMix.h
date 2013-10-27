/*
 * Payload definition(s): Chamber Gas Mix (RPCObGasMix)
 *
 *  $Date: 2009/12/13 09:56:28 $
 *  $Revision: 1.1 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#ifndef RPCObGasMix_h
#define RPCObGasMix_h
#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>

class RPCObGasMix {
    public:
      struct Item {
        int unixtime;
        float gas1; // IC4H10
        float gas2; // C2H2F4
        float gas3; // SF6
      
      COND_SERIALIZABLE;
};
    RPCObGasMix(){}
    virtual ~RPCObGasMix(){}
    std::vector<Item>  ObGasMix_rpc;
   
   COND_SERIALIZABLE;
};

#endif

