/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2004                                                      *
*                                                                              *
*******************************************************************************/
#include "L1Trigger/RPCTrigger/src/L1RpcTriggerBoard.h"
#ifdef _GRAB_MUONS
#include "PactAnalysis/L1RpcMuonsGrabber.h"
#endif

//---------------------------------------------------------------------------
L1RpcTBMuonsVec L1RpcTriggerBoard::RunTBGB() { //4 muons or empty vector
  if(PacsMuonsVec.size() == 0)
    return L1RpcTBMuonsVec();

  #ifdef _GRAB_MUONS
    L1RpcMuonsGrabber::Instance()->StorePacMuons(PacsMuonsVec);
  #endif

  L1RpcTBMuonsVec2 gbMuons(L1RpcConst::TOWERS_ON_TB_CNT, L1RpcTBMuonsVec());
  for(unsigned int iMu = 0; iMu < PacsMuonsVec.size(); iMu++) {
    int tbTower = TriggerConfig->GetTowerNumOnTb(PacsMuonsVec[iMu].GetConeCrdnts() );

    if(gbMuons[tbTower].size() == 0)
      gbMuons[tbTower].assign(L1RpcConst::SEGMENTS_IN_SECTOR_CNT, L1RpcTBMuon());

    gbMuons[tbTower][PacsMuonsVec[iMu].GetLogSegment()] = PacsMuonsVec[iMu];
  }

  PacsMuonsVec.clear();
  return TBGhostBuster->Run(gbMuons);
}
//---------------------------------------------------------------------------
L1RpcTriggerBoard::L1RpcTriggerBoard(L1RpcTBGhostBuster* tbGhostBuster,
                L1RpcTriggerConfiguration* triggerConfig,
                int tbNum) {
    TBGhostBuster = tbGhostBuster;
    TriggerConfig = triggerConfig;
    TBNumber = tbNum;
}

//---------------------------------------------------------------------------
bool L1RpcTriggerBoard::RunCone(const L1RpcLogCone& cone)  {

    L1RpcTBMuon tbMuon(TriggerConfig->GetPac(cone.GetConeCrdnts())->Run(cone) );
    
    if(tbMuon.GetCode() > 0) {
        PacsMuonsVec.push_back(tbMuon);
        if (TriggerConfig->GetDebugLevel()!=0){
#ifndef _STAND_ALONE
	  LogDebug("RPCHwDebug") << "GB 0"
			         << tbMuon.printDebugInfo(TriggerConfig->GetDebugLevel());
#else
	  std::cout << "GB 0"
  	  	    << tbMuon.printDebugInfo(TriggerConfig->GetDebugLevel())
		    << std::endl;
#endif //_STAND_ALONE
        }
      return true;
    }
    else
      return false;
      
}
