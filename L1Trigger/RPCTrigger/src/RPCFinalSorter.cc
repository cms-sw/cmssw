//---------------------------------------------------------------------------
#include "L1Trigger/RPCTrigger/interface/RPCFinalSorter.h"

#include "L1Trigger/RPCTrigger/interface/RPCConst.h"
#include "L1Trigger/RPCTrigger/interface/MuonsGrabber.h"
#include <algorithm>
//---------------------------------------------------------------------------
#include <set>
using namespace std;

/**
 *
 * Defualt constructor
 *
*/
RPCFinalSorter::RPCFinalSorter(RPCTriggerConfiguration* triggerConfig) {
  
  m_TrigCnfg = triggerConfig;
  m_GBOutputMuons.assign(2, L1RpcTBMuonsVec());
  
}
/** 
 * Converts m_tower number (eta addr) from continous (0 - 32, m_tower 0 = 16)
 * to 2'complement.
 * @return 4 munons from barrel (m_GBOutputMuons[0]),
 * and 4 from endcaps (m_GBOutputMuons[1]).
*/
L1RpcTBMuonsVec2 RPCFinalSorter::run(L1RpcTBMuonsVec2 &tcsMuonsVec2) {
  
  //m_GBOutputMuons[0].clear();
  //m_GBOutputMuons[1].clear();
    
  runFinalSorter(tcsMuonsVec2); // modyfies  tcsMuonsVec2


  if (m_TrigCnfg->getDebugLevel()!=0){
    for (unsigned  int region = 0; region < tcsMuonsVec2.size(); ++region){
        for (unsigned  int iTB = 0; iTB < tcsMuonsVec2[region].size(); iTB++){
#ifndef _STAND_ALONE
          //  LogDebug("RPCHwDebug")<<"GB 4" << region
	  //      << "0 " << iTB << " "
          //      << tcsMuonsVec2[region][iTB].printDebugInfo(m_TrigCnfg->getDebugLevel());
          MuonsGrabber::Instance().addMuon(tcsMuonsVec2[region][iTB], 4, region, 0, iTB);  
#else
            std::cout<<"GB 4" << region 
	        << "0 " << iTB << " "
                << tcsMuonsVec2[region][iTB].printDebugInfo(m_TrigCnfg->getDebugLevel())
                << std::endl;
#endif
        }
    }
  }

  return tcsMuonsVec2;
}
/*
 *
 *
 *
*/
void RPCFinalSorter::runFinalSorter(L1RpcTBMuonsVec2 &finalMuons) {
//---------sorting-----------------------------------------
  sort(finalMuons[0].begin(), finalMuons[0].end(), RPCTBMuon::TMuonMore()) ;
  sort(finalMuons[1].begin(), finalMuons[1].end(), RPCTBMuon::TMuonMore()) ;

//-------setting size to m_GBETA_OUT_MUONS_CNT----------------
  while(finalMuons[0].size() > RPCConst::m_FINAL_OUT_MUONS_CNT) {
    finalMuons[0].pop_back();
    finalMuons[1].pop_back();
  }

//-----setting 2'complement EtaAddr
  for(unsigned int iMu = 0; iMu < finalMuons[0].size(); iMu++) {
    // 10 oct 2006 - moved to RPCTCGhostBusterSorter::run 
    //finalMuons[0][iMu].setEtaAddr(m_TrigCnfg->towNum2TowNum2Comp(finalMuons[0][iMu].getEtaAddr()));
    //finalMuons[1][iMu].setEtaAddr(m_TrigCnfg->towNum2TowNum2Comp(finalMuons[1][iMu].getEtaAddr()));
    
    // 9 July 2007 - temporarly moved to HS
    //finalMuons[0][iMu].setPhiAddr(finalMuons[0][iMu].getContinSegmAddr());
    //finalMuons[1][iMu].setPhiAddr(finalMuons[1][iMu].getContinSegmAddr());   
  }  
}
