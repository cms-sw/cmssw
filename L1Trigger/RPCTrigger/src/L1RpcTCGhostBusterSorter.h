/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2004                                                      *
*                                                                              *
*******************************************************************************/
#ifndef L1RpcTCGhostBusterSorterH
#define L1RpcTCGhostBusterSorterH
/** \class L1RpcTCGhostBusterSorter
  * Peformes the Trigger Crate Ghost Buster and sorter algorithm.
  * Because the class does not keep any data and GB is the same for every TC,
  * there might be one and the same object of this class for all TCs.
  * \author Karol Bunkowski (Warsaw)
  */

#include "L1Trigger/RPCTrigger/src/L1RpcTBMuon.h"
#include "L1Trigger/RPCTrigger/src/L1RpcTriggerConfiguration.h"

//---------------------------------------------------------------------------
class L1RpcTCGhostBusterSorter {
public:
  L1RpcTCGhostBusterSorter(L1RpcTriggerConfiguration* triggerConfig);

  /** Peformes the Trigger Crate Ghost Buster and sorter algorithm -
    * in eta between TB of one TC.
    * Coverts muons etaAddr from 2bit tow num on TB (0-2 or 0-3)
    * to continous tower number (etaAddr) (0 - 32, tower0 = 16)
    * @return always 4 muons*/
  L1RpcTBMuonsVec Run(L1RpcTBMuonsVec2 &tbMuonsVec);
private:

  L1RpcTriggerConfiguration* TriggerConfig;
};
#endif
