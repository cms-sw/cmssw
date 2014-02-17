/*
 * Payload definition(s): Chamber Gas Mix (RPCObGasMix)
 *
 *  $Date: 2009/12/14 16:00:11 $
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
        float gas1; // IC4H10
        float gas2; // C2H2F4
        float gas3; // SF6
      };
    RPCObGasMix(){}
    virtual ~RPCObGasMix(){}
    std::vector<Item>  ObGasMix_rpc;
   };

#endif

