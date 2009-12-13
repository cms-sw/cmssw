/*
 * Payload definition(s): Chamber Gas Mix (RPCObGasMix)
 *
 *  $Date: 2009/11/16 13:00:18 $
 *  $Revision: 1.2 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#ifndef RPCObGasMix_h
#define RPCObGasMix_h
#include <vector>

class RPCObGasMix {
    public:
      struct Item {
        int unixtime;
        float gas1; 
        float gas2; 
        float gas3;
      };
    RPCObGasMix(){}
    virtual ~RPCObGasMix(){}
    std::vector<Item>  ObGasMix_rpc;
   };

#endif

