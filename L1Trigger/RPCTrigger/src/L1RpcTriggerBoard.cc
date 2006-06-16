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

  L1RpcTBMuonsVec2 gbMuons(RPCParam::TOWERS_ON_TB_CNT, L1RpcTBMuonsVec());
  for(int iMu = 0; iMu < PacsMuonsVec.size(); iMu++) {
    int tbTower = TriggerConfig->GetTowerNumOnTb(PacsMuonsVec[iMu].GetConeCrdnts() );

    if(gbMuons[tbTower].size() == 0)
      gbMuons[tbTower].assign(RPCParam::SEGMENTS_IN_SECTOR_CNT, L1RpcTBMuon());

    gbMuons[tbTower][PacsMuonsVec[iMu].GetLogSegment()] = PacsMuonsVec[iMu];
  }

  PacsMuonsVec.clear();
  return TBGhostBuster->Run(gbMuons);
}
