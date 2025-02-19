#ifndef L1Trigger_RPCHalfSorter_h
#define L1Trigger_RPCHalfSorter_h

/** \class RPCHalfSorter
  * Emulates HalfSorter (currently both of them ;-) )
  * \author Tomasz Fruboes (based on code by Karol Bunkowski) 
  */

#include "L1Trigger/RPCTrigger/interface/RPCTBMuon.h"
#include "L1Trigger/RPCTrigger/interface/RPCTriggerConfiguration.h"

#include <FWCore/Framework/interface/ESHandle.h>
#include "CondFormats/L1TObjects/interface/L1RPCHsbConfig.h"
//---------------------------------------------------------------------------

class RPCHalfSorter {
public:
  
  RPCHalfSorter(RPCTriggerConfiguration* triggerConfig);

  L1RpcTBMuonsVec2 runHalf(L1RpcTBMuonsVec2 &tcsMuonsVec2);

  void maskHSBInput(L1RpcTBMuonsVec & newVec, int mask);

  L1RpcTBMuonsVec2 run(L1RpcTBMuonsVec2 &tcsMuonsVec2,  edm::ESHandle<L1RPCHsbConfig> hsbConf );

private:
  L1RpcTBMuonsVec2 m_GBOutputMuons;

  RPCTriggerConfiguration* m_TrigCnfg;
  //m_GBOutputMuons[be][iMU] , be = 0 = barrel; be = 1 = endcap
};
#endif
