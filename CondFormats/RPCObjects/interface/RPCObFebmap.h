/*
 * Payload definition(s): Feb thresholds, temperatures, voltages and noises (RPCObFebmap)
 *
 *  $Date: 2009/11/16 12:59:31 $
 *  $Revision: 1.4 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */


#ifndef RPCObFebmap_h
#define RPCObFebmap_h
#include <vector>


class RPCObFebmap {
    public:
      struct Feb_Item {
        int   dpid;
        float thr1;
        float thr2;
        float thr3;
        float thr4;
        float vmon1;
        float vmon2;
        float vmon3;
        float vmon4;
        float temp1;
        float temp2;
        int   day;
        int   time;
	int   noise1;
        int   noise2;
        int   noise3;
        int   noise4;
    };
    RPCObFebmap(){}
    virtual ~RPCObFebmap(){}
    std::vector<Feb_Item> ObFebMap_rpc;
   };

#endif

