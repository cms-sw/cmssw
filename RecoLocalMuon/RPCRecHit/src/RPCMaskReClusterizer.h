#ifndef RecoLocalMuon_RPCMaskReClusterizer_h
#define RecoLocalMuon_RPCMaskReClusterizer_h

/** \Class RPCMaskReClusterizer
 *  $Date: 2008/08/12 16:06:39  $
 *  $Revision: 1.0 $
 *  \author J.C. Sanabria -- UniAndes, Bogota
 */

#include <bitset>

#include "RPCCluster.h"
#include "RPCClusterizer.h"
#include "RPCClusterContainer.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"


const int SIZE=96;

typedef std::bitset<SIZE> RollMask;


class RPCMaskReClusterizer 
{
 public :

   RPCMaskReClusterizer();
   ~RPCMaskReClusterizer();
   RPCClusterContainer doAction(const RPCDetId& ,RPCClusterContainer& , const RollMask& );
   int get(const RollMask& ,int );

};

#endif
