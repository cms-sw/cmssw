#ifndef L1Trigger_RPCFinalSorter_h
#define L1Trigger_RPCFinalSorter_h

/** \class RPCFinalSorter
  * Emulates FinalSorter 
  * \author Karol Bunkowski (Warsaw)
  */

#include "L1Trigger/RPCTrigger/interface/RPCTBMuon.h"
#include "L1Trigger/RPCTrigger/interface/RPCTriggerConfiguration.h"
//---------------------------------------------------------------------------

class RPCFinalSorter {
public:
  
  RPCFinalSorter(RPCTriggerConfiguration* triggerConfig);

  L1RpcTBMuonsVec2 runHalf(L1RpcTBMuonsVec2 &tcsMuonsVec2);

  //m_GBOutputMuons[be][iMU] , be = 0 = barrel; be = 1 = endcap
  //on input m_tower number (eta addr) continous (0 - 32, m_tower 0 = 16)
  //on output m_tower number (eta addr) 2'complement

  void runFinalSorter(L1RpcTBMuonsVec2 &finalMuons);

  //L1RpcTBMuonsVec2 run(L1RpcTBMuonsVec2 &tcsMuonsVec2);
  L1RpcTBMuonsVec2 run(L1RpcTBMuonsVec2 &tcsMuonsVec2);

private:
  L1RpcTBMuonsVec2 m_GBOutputMuons;

  RPCTriggerConfiguration* m_TrigCnfg;
  //m_GBOutputMuons[be][iMU] , be = 0 = barrel; be = 1 = endcap
};
#endif
