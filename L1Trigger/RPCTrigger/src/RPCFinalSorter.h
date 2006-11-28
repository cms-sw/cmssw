//---------------------------------------------------------------------------

#ifndef L1RpcFinalSorterH
#define L1RpcFinalSorterH

/** \class RPCFinalSorter
  * Peformes the m_FinalSorter Ghost Buster and sorter algorithms.
  * Because the class does not keep any data and GB is the same for every TB,
  * there might be one and the same object of this class for all TBs.
  * \author Karol Bunkowski (Warsaw)
  */

#include "L1Trigger/RPCTrigger/src/RPCTBMuon.h"
#include "L1Trigger/RPCTrigger/src/RPCTriggerConfiguration.h"
//---------------------------------------------------------------------------

class RPCFinalSorter {
public:
  
  RPCFinalSorter(RPCTriggerConfiguration* triggerConfig);

  L1RpcTBMuonsVec2 runHalf(L1RpcTBMuonsVec2 &tcsMuonsVec2);

  //m_GBOutputMuons[be][iMU] , be = 0 = barrel; be = 1 = endcap
  //on input m_tower number (eta addr) continous (0 - 32, m_tower 0 = 16)
  //on output m_tower number (eta addr) 2'complement

  void runFinalSorter(L1RpcTBMuonsVec2 &finalMuons);

  L1RpcTBMuonsVec2 run(L1RpcTBMuonsVec2 &tcsMuonsVec2);

private:
  L1RpcTBMuonsVec2 m_GBOutputMuons;

  RPCTriggerConfiguration* m_TrigCnfg;
  //m_GBOutputMuons[be][iMU] , be = 0 = barrel; be = 1 = endcap
};
#endif
