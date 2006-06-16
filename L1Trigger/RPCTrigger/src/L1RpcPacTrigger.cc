/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2004                                                      *
*                                                                              *
*******************************************************************************/
#include "L1Trigger/RPCTrigger/src/L1RpcPacTrigger.h"
#ifdef _GRAB_MUONS
#include "PactAnalysis/L1RpcMuonsGrabber.h"
#endif

L1RpcPacTrigger::L1RpcPacTrigger(L1RpcTriggerConfiguration* triggerConfig):
  FinalSorter(triggerConfig), TCGhostBusterSorter(triggerConfig), TBGhostBuster() {
  TrigCnfg = triggerConfig;
  for(int iTC = 0; iTC < TrigCnfg->GetTCsCnt(); iTC++) {
    TriggerCratesVec.push_back(L1RpcTriggerCrate(&TCGhostBusterSorter, &TBGhostBuster, TrigCnfg, iTC) );
  }
}

////////////////////////////////////////////////////////////////////////////////////

L1RpcTBMuonsVec2 L1RpcPacTrigger::RunEvent(const L1RpcLogConesVec& logConesVec) {
  GBFinalMuons.clear();
  for(int iLC = 0; iLC < logConesVec.size(); iLC++) {
    if(logConesVec[iLC].GetFiredPlanesCnt() >= 3)
      TriggerCratesVec[TrigCnfg->GetTCNum(logConesVec[iLC].GetConeCrdnts())].RunCone(logConesVec[iLC]);
  }

  L1RpcTBMuonsVec2 tcsMuonsVec2;
  for(int iTC = 0; iTC < TriggerCratesVec.size(); iTC++) {
    tcsMuonsVec2.push_back(TriggerCratesVec[iTC].RunTCGBSorter() );
  }

  #ifdef _GRAB_MUONS
    L1RpcMuonsGrabber::Instance()->StoreFinalGBMuons(tcsMuonsVec2);
  #endif

  GBFinalMuons = FinalSorter.Run(tcsMuonsVec2);

  #ifdef _GRAB_MUONS
    L1RpcMuonsGrabber::Instance()->StoreAnswers(GBFinalMuons);
  #endif

  return GBFinalMuons;
};

////////////////////////////////////////////////////////////////////////////////////

L1RpcTBMuonsVec L1RpcPacTrigger::GetNotEmptyMuons()  {
  L1RpcTBMuonsVec notEmptyMuonsVec;
  for(int iMu = 0; iMu < GBFinalMuons[0].size(); iMu++)
    if(GBFinalMuons[0][iMu].GetCode() != 0)
      notEmptyMuonsVec.push_back(GBFinalMuons[0][iMu]);

  for(int iMu = 0; iMu < GBFinalMuons[1].size(); iMu++)
    if(GBFinalMuons[1][iMu].GetCode() != 0)
      notEmptyMuonsVec.push_back(GBFinalMuons[1][iMu]);

  return notEmptyMuonsVec;
}
