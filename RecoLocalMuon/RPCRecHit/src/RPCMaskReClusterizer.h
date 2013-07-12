#ifndef RecoLocalMuon_RPCMaskReClusterizer_h
#define RecoLocalMuon_RPCMaskReClusterizer_h

/** \Class RPCMaskReClusterizer
 *  $Date: 2008/10/14 09:18:20 $
 *  $Revision: 1.1 $
 *  \author J.C. Sanabria -- UniAndes, Bogota
 */

#include "RPCRollMask.h"

#include "RPCCluster.h"
#include "RPCClusterizer.h"
#include "RPCClusterContainer.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"


class RPCMaskReClusterizer 
{
 public :

   RPCMaskReClusterizer();
   ~RPCMaskReClusterizer();
   RPCClusterContainer doAction(const RPCDetId& ,RPCClusterContainer& , const RollMask& );
   int get(const RollMask& ,int );

};

#endif
