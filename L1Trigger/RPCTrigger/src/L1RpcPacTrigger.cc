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

 if (TrigCnfg->GetDebugLevel()!=0){
#ifdef _STAND_ALONE
   std::cout << "---TBMuons in new event" << std::endl;
#else
   LogDebug("RPCHwDebug") << "---TBMuons in new event";
#endif // _STAND_ALONE
 }
  for(unsigned int iLC = 0; iLC < logConesVec.size(); iLC++) {
    if(logConesVec[iLC].GetFiredPlanesCnt() >= 3) {
      TriggerCratesVec[TrigCnfg->GetTCNum(logConesVec[iLC].GetConeCrdnts())].RunCone(logConesVec[iLC]);
    }
  }

  L1RpcTBMuonsVec2 tcsMuonsVec2;
  for(unsigned int iTC = 0; iTC < TriggerCratesVec.size(); iTC++) {
    tcsMuonsVec2.push_back(TriggerCratesVec[iTC].RunTCGBSorter() );
  }
      
  if (TrigCnfg->GetDebugLevel()!=0){
    for (unsigned  int iTC = 0; iTC < tcsMuonsVec2.size(); iTC++){
        for (unsigned  int iTB = 0; iTB < tcsMuonsVec2[iTC].size(); iTB++){
#ifdef _STAND_ALONE
	  std::cout << "GB 2 " <<tcsMuonsVec2[iTC][iTB].printDebugInfo(TrigCnfg->GetDebugLevel()) << std::endl;
#else
	  LogDebug("RPCHwDebug") << "GB 2 " <<tcsMuonsVec2[iTC][iTB].printDebugInfo(TrigCnfg->GetDebugLevel());
#endif // _STAND_ALONE
        }
    }
  }

  GBFinalMuons = FinalSorter.Run(tcsMuonsVec2);
  /* // Moved to FinalSorter to have HS and FS info
  if (TrigCnfg->GetDebugLevel()!=0){
     // iterate over GBFinalMuons and call printDebug()
    for (unsigned  int iTC = 0; iTC < GBFinalMuons.size(); iTC++){
        for (unsigned  int iTB = 0; iTB < GBFinalMuons[iTC].size(); iTB++){
#ifdef _STAND_ALONE
	  std::cout <<"GB 3 "<<GBFinalMuons[iTC][iTB].printDebugInfo(TrigCnfg->GetDebugLevel())<< std::endl;
#else
	  LogDebug("RPCHwDebug") <<"GB 3 "<<GBFinalMuons[iTC][iTB].printDebugInfo(TrigCnfg->GetDebugLevel());

#endif // _STAND_ALONE
        }
    }                               
  }
  */
  /*
  #ifdef _GRAB_MUONS
    L1RpcMuonsGrabber::Instance()->StoreAnswers(GBFinalMuons);
  #endif
  */
  
#ifdef GETCONES
  bool foundMuons = false;
  L1RpcTBMuonsVec bMuons = GBFinalMuons[0];
  L1RpcTBMuonsVec fMuons = GBFinalMuons[1];

  std::cout<< "------------" << std::endl;
  for (unsigned int i = 0; i < bMuons.size(); i++){
    if (bMuons[i].GetPtCode()!=0){ 
       /*
       std::cout<< "ptcode " << bMuons[i].GetPtCode() 
                << " t " <<  bMuons[i].GetTower()
                << " sec " <<  bMuons[i].GetLogSector()
                << " seg " <<  bMuons[i].GetLogSegment()
                << std::endl;*/
       foundMuons = true;
   }
  }
  for (unsigned int i = 0; i < fMuons.size(); i++){
    if (fMuons[i].GetPtCode()!=0){
       /*std::cout<< "ptcode " << fMuons[i].GetPtCode()
                << " t " <<  fMuons[i].GetTower()
                << " sec " <<  fMuons[i].GetLogSector()
                << " seg " <<  fMuons[i].GetLogSegment()
                << std::endl;*/
       foundMuons = true;
       }
  }
  if (!foundMuons){
    for(unsigned int iLC = 0; iLC < logConesVec.size(); iLC++) {
      if(logConesVec[iLC].GetFiredPlanesCnt() >= 3) {
        std::cout<< logConesVec[iLC].toString();
      }
    }
  }
#endif



  return GBFinalMuons;
};

////////////////////////////////////////////////////////////////////////////////////

L1RpcTBMuonsVec L1RpcPacTrigger::GetNotEmptyMuons()  {
  L1RpcTBMuonsVec notEmptyMuonsVec;
  for(unsigned int iMu = 0; iMu < GBFinalMuons[0].size(); iMu++)
    if(GBFinalMuons[0][iMu].GetCode() != 0)
      notEmptyMuonsVec.push_back(GBFinalMuons[0][iMu]);

  for(unsigned int iMu = 0; iMu < GBFinalMuons[1].size(); iMu++)
    if(GBFinalMuons[1][iMu].GetCode() != 0)
      notEmptyMuonsVec.push_back(GBFinalMuons[1][iMu]);

  return notEmptyMuonsVec;
}
