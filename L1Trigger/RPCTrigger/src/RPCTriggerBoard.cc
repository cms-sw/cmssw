/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2004                                                      *
*                                                                              *
*******************************************************************************/
#include "L1Trigger/RPCTrigger/src/RPCTriggerBoard.h"
#ifdef _GRAB_MUONS
#include "PactAnalysis/L1RpcMuonsGrabber.h"
#endif

//---------------------------------------------------------------------------
L1RpcTBMuonsVec RPCTriggerBoard::runTBGB() { //4 muons or empty vector
  if(m_PacsMuonsVec.size() == 0)
    return L1RpcTBMuonsVec();

  #ifdef _GRAB_MUONS
    L1RpcMuonsGrabber::Instance()->StorePacMuons(m_PacsMuonsVec);
  #endif

  L1RpcTBMuonsVec2 gbMuons(RPCConst::m_TOWERS_ON_TB_CNT, L1RpcTBMuonsVec());
  for(unsigned int iMu = 0; iMu < m_PacsMuonsVec.size(); iMu++) {
    int tbTower = m_TriggerConfig->getTowerNumOnTb(m_PacsMuonsVec[iMu].getConeCrdnts() );

    if(gbMuons[tbTower].size() == 0)
      gbMuons[tbTower].assign(RPCConst::m_SEGMENTS_IN_SECTOR_CNT, RPCTBMuon());

    gbMuons[tbTower][m_PacsMuonsVec[iMu].getLogSegment()] = m_PacsMuonsVec[iMu];
  }

  m_PacsMuonsVec.clear();
  return m_TBGhostBuster->run(gbMuons);
}
//---------------------------------------------------------------------------
RPCTriggerBoard::RPCTriggerBoard(RPCTBGhostBuster* tbGhostBuster,
                RPCTriggerConfiguration* triggerConfig,
                int tbNum) {
    m_TBGhostBuster = tbGhostBuster;
    m_TriggerConfig = triggerConfig;
    m_TBNumber = tbNum;
}

//---------------------------------------------------------------------------
bool RPCTriggerBoard::runCone(const RPCLogCone& cone)  {

    RPCTBMuon tbMuon(m_TriggerConfig->getPac(cone.getConeCrdnts())->run(cone) );
    
    if(tbMuon.getCode() > 0) {
        m_PacsMuonsVec.push_back(tbMuon);
        if (m_TriggerConfig->getDebugLevel()!=0){
#ifndef _STAND_ALONE
	  LogDebug("RPCHwDebug") << "GB 0"
			         << tbMuon.printDebugInfo(m_TriggerConfig->getDebugLevel());
#else
	  std::cout << "GB 0"
  	  	    << tbMuon.printDebugInfo(m_TriggerConfig->getDebugLevel())
		    << std::endl;
#endif //_STAND_ALONE
        }
      return true;
    }
    else
      return false;
      
}
