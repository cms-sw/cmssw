#ifndef L1Trigger_RPCTCGhostBusterSorter_h
#define L1Trigger_RPCTCGhostBusterSorter_h
/** \class RPCTCGhostBusterSorter
  * Peformes the Trigger Crate Ghost Buster and sorter algorithm.
  * Because the class does not keep any data and GB is the same for every TC,
  * there might be one and the same object of this class for all TCs.
  * \author Karol Bunkowski (Warsaw)
  */

#include "L1Trigger/RPCTrigger/interface/RPCTBMuon.h"
#include "L1Trigger/RPCTrigger/interface/RPCTriggerConfiguration.h"

//---------------------------------------------------------------------------
class RPCTCGhostBusterSorter {
public:
  RPCTCGhostBusterSorter(RPCTriggerConfiguration* triggerConfig);
  
  /** Peformes the Trigger Crate Ghost Buster and sorter algorithm -
    * in eta between TB of one TC.
    * Coverts muons etaAddr from 2bit tow num on TB (0-2 or 0-3)
    * to continous m_tower number (etaAddr) (0 - 32, tower0 = 16)
    * @return always 4 muons*/
  L1RpcTBMuonsVec run(L1RpcTBMuonsVec2 &tbMuonsVec);
private:

  RPCTriggerConfiguration* m_TriggerConfig;
};
#endif
