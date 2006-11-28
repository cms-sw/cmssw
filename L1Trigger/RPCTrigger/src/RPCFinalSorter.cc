//---------------------------------------------------------------------------
#include "L1Trigger/RPCTrigger/src/RPCFinalSorter.h"

#include "L1Trigger/RPCTrigger/src/RPCConst.h"
#ifdef _GRAB_MUONS
#include "PactAnalysis/L1RpcMuonsGrabber.h"
#endif
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
  
  m_GBOutputMuons.assign(2, L1RpcTBMuonsVec() );
  
}
/** 
 *
 * Runs GB algorithm for half of the detector - 6 TC (sectors).
 * @return 4 munons from barrel (m_GBOutputMuons[0]),
 * and 4 from endcaps (m_GBOutputMuons[1]).
*/
L1RpcTBMuonsVec2 RPCFinalSorter::runHalf(L1RpcTBMuonsVec2 &tcsMuonsVec2) {
  //1+6+1 TC, wazne sa tylko te 6 w srodku
  for(unsigned int iTC = 0; iTC < tcsMuonsVec2.size()-1; iTC++) {
    for(unsigned int iMu = 0; iMu < tcsMuonsVec2[iTC].size(); iMu++) {
      if(tcsMuonsVec2[iTC][iMu].getCode() == 0)
        continue;
      if(tcsMuonsVec2[iTC][iMu].gBDataKilledLast() ) {
        for(unsigned int iMuN = 0; iMuN < tcsMuonsVec2[iTC+1].size(); iMuN++) {
          if(tcsMuonsVec2[iTC+1][iMuN].getCode() == 0)
            continue;
          if(tcsMuonsVec2[iTC+1][iMuN].gBDataKilledFirst() ) {
            if( abs(tcsMuonsVec2[iTC][iMu].getEtaAddr() -
                    tcsMuonsVec2[iTC+1][iMuN].getEtaAddr() ) <= 1 )
              if(tcsMuonsVec2[iTC][iMu].getCode() <= tcsMuonsVec2[iTC+1][iMuN].getCode() ) {
                if(tcsMuonsVec2[iTC][iMu].getSegmentAddr() == RPCConst::m_SEGMENTS_IN_SECTOR_CNT-1)
                  tcsMuonsVec2[iTC][iMu].kill();
              }    
              else
                tcsMuonsVec2[iTC+1][iMuN].kill();
          }
        }
      }
    }
  }

  L1RpcTBMuonsVec outputBarrelMuons;
  L1RpcTBMuonsVec outputEndcapMuons;
  
  for(unsigned int iTC = 1; iTC < tcsMuonsVec2.size()-1; iTC++)
    for(unsigned int iMu = 0; iMu < tcsMuonsVec2[iTC].size(); iMu++)
      if(tcsMuonsVec2[iTC][iMu].isLive() ) {
        if(abs(16 - tcsMuonsVec2[iTC][iMu].getEtaAddr()) <=7 )
          outputBarrelMuons.push_back(tcsMuonsVec2[iTC][iMu]);
        else
          outputEndcapMuons.push_back(tcsMuonsVec2[iTC][iMu]);
      }
      
  sort(outputBarrelMuons.begin(), outputBarrelMuons.end(), RPCTBMuon::TMuonMore());
  sort(outputEndcapMuons.begin(), outputEndcapMuons.end(), RPCTBMuon::TMuonMore());

  while(outputBarrelMuons.size() < RPCConst::m_FINAL_OUT_MUONS_CNT)
    outputBarrelMuons.push_back(RPCTBMuon());
  while(outputBarrelMuons.size() > RPCConst::m_FINAL_OUT_MUONS_CNT)
    outputBarrelMuons.pop_back();
  
  while(outputEndcapMuons.size() < RPCConst::m_FINAL_OUT_MUONS_CNT)
    outputEndcapMuons.push_back(RPCTBMuon());
  while(outputEndcapMuons.size() > RPCConst::m_FINAL_OUT_MUONS_CNT)
    outputEndcapMuons.pop_back();

  m_GBOutputMuons[0].insert(m_GBOutputMuons[0].end(), outputBarrelMuons.begin(), outputBarrelMuons.end());
  m_GBOutputMuons[1].insert(m_GBOutputMuons[1].end(), outputEndcapMuons.begin(), outputEndcapMuons.end());
  return m_GBOutputMuons;
}
/** 
 * Runs runHalf() for 2 detecors parts.
 * Converts m_tower number (eta addr) from continous (0 - 32, m_tower 0 = 16)
 * to 2'complement.
 * @return 4 munons from barrel (m_GBOutputMuons[0]),
 * and 4 from endcaps (m_GBOutputMuons[1]).
*/
L1RpcTBMuonsVec2 RPCFinalSorter::run(L1RpcTBMuonsVec2 &tcsMuonsVec2) {
  
  m_GBOutputMuons[0].clear();
  m_GBOutputMuons[1].clear();

  L1RpcTBMuonsVec2 firstHalfTcsMuonsVec2;

  firstHalfTcsMuonsVec2.push_back(tcsMuonsVec2[m_TrigCnfg->getTCsCnt()-1]);
  for(int iTC = 0; iTC < m_TrigCnfg->getTCsCnt()/2 +1; iTC++) {
    for(unsigned int iMu = 0; iMu < tcsMuonsVec2[iTC].size(); iMu++)
      tcsMuonsVec2[iTC][iMu].setSectorAddr(iTC);
    firstHalfTcsMuonsVec2.push_back(tcsMuonsVec2[iTC]);
  }

  runHalf(firstHalfTcsMuonsVec2);
  
  L1RpcTBMuonsVec2 secondHalfTcsMuonsVec2;
  for(int iTC = m_TrigCnfg->getTCsCnt()/2-1; iTC < m_TrigCnfg->getTCsCnt(); iTC++) {
    for(unsigned int iMu = 0; iMu < tcsMuonsVec2[iTC].size(); iMu++)
      tcsMuonsVec2[iTC][iMu].setSectorAddr(iTC);
    secondHalfTcsMuonsVec2.push_back(tcsMuonsVec2[iTC]);
  }
  secondHalfTcsMuonsVec2.push_back(tcsMuonsVec2[0]);

  runHalf(secondHalfTcsMuonsVec2);
  // Debug
  if (m_TrigCnfg->getDebugLevel()!=0){
    for (unsigned  int iTC = 0; iTC < m_GBOutputMuons.size(); iTC++){
        for (unsigned  int iTB = 0; iTB < m_GBOutputMuons[iTC].size(); iTB++){
#ifndef _STAND_ALONE
            LogDebug("RPCHwDebug")<<"GB 3 "<< m_GBOutputMuons[iTC][iTB].printDebugInfo(m_TrigCnfg->getDebugLevel());
#else
            std::cout <<"GB 3 "<< m_GBOutputMuons[iTC][iTB].printDebugInfo(m_TrigCnfg->getDebugLevel()) << std::endl;
#endif 
        }
    }
  }

  runFinalSorter(m_GBOutputMuons);

  if (m_TrigCnfg->getDebugLevel()!=0){
    for (unsigned  int iTC = 0; iTC < m_GBOutputMuons.size(); iTC++){
        for (unsigned  int iTB = 0; iTB < m_GBOutputMuons[iTC].size(); iTB++){
#ifndef _STAND_ALONE
            LogDebug("RPCHwDebug")<<"GB 4 "<< m_GBOutputMuons[iTC][iTB].printDebugInfo(m_TrigCnfg->getDebugLevel());
#else
            std::cout<<"GB 4 "<< m_GBOutputMuons[iTC][iTB].printDebugInfo(m_TrigCnfg->getDebugLevel()) << std::endl;
#endif
        }
    }
  }

  return m_GBOutputMuons;
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
  for(unsigned int iMu = 0; iMu < m_GBOutputMuons[0].size(); iMu++) {
    // 10 oct 2006 - moved to RPCTCGhostBusterSorter::run 
    //finalMuons[0][iMu].setEtaAddr(m_TrigCnfg->towNum2TowNum2Comp(finalMuons[0][iMu].getEtaAddr()));
    //finalMuons[1][iMu].setEtaAddr(m_TrigCnfg->towNum2TowNum2Comp(finalMuons[1][iMu].getEtaAddr()));
   
    finalMuons[0][iMu].setPhiAddr(finalMuons[0][iMu].getContinSegmAddr());
    finalMuons[1][iMu].setPhiAddr(finalMuons[1][iMu].getContinSegmAddr());   
  }  
}
