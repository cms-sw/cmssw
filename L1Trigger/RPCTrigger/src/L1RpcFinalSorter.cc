//---------------------------------------------------------------------------
#include "L1Trigger/RPCTrigger/src/L1RpcFinalSorter.h"

#include "L1Trigger/RPCTrigger/src/L1RpcConst.h"
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
L1RpcFinalSorter::L1RpcFinalSorter(L1RpcTriggerConfiguration* triggerConfig) {
  
  TrigCnfg = triggerConfig;
  
  GBOutputMuons.assign(2, L1RpcTBMuonsVec() );
  
}
/** 
 *
 * Runs GB algorithm for half of the detector - 6 TC (sectors).
 * @return 4 munons from barrel (GBOutputMuons[0]),
 * and 4 from endcaps (GBOutputMuons[1]).
*/
L1RpcTBMuonsVec2 L1RpcFinalSorter::RunHalf(L1RpcTBMuonsVec2 &tcsMuonsVec2) {
  //1+6+1 TC, wazne sa tylko te 6 w srodku
  for(unsigned int iTC = 0; iTC < tcsMuonsVec2.size()-1; iTC++) {
    for(unsigned int iMu = 0; iMu < tcsMuonsVec2[iTC].size(); iMu++) {
      if(tcsMuonsVec2[iTC][iMu].GetCode() == 0)
        continue;
      if(tcsMuonsVec2[iTC][iMu].GBDataKilledLast() ) {
        for(unsigned int iMuN = 0; iMuN < tcsMuonsVec2[iTC+1].size(); iMuN++) {
          if(tcsMuonsVec2[iTC+1][iMuN].GetCode() == 0)
            continue;
          if(tcsMuonsVec2[iTC+1][iMuN].GBDataKilledFirst() ) {
            if( abs(tcsMuonsVec2[iTC][iMu].GetEtaAddr() -
                    tcsMuonsVec2[iTC+1][iMuN].GetEtaAddr() ) <= 1 )
              if(tcsMuonsVec2[iTC][iMu].GetCode() <= tcsMuonsVec2[iTC+1][iMuN].GetCode() ) {
                if(tcsMuonsVec2[iTC][iMu].GetSegmentAddr() == L1RpcConst::SEGMENTS_IN_SECTOR_CNT-1)
                  tcsMuonsVec2[iTC][iMu].Kill();
              }    
              else
                tcsMuonsVec2[iTC+1][iMuN].Kill();
          }
        }
      }
    }
  }

  L1RpcTBMuonsVec outputBarrelMuons;
  L1RpcTBMuonsVec outputEndcapMuons;
  
  for(unsigned int iTC = 1; iTC < tcsMuonsVec2.size()-1; iTC++)
    for(unsigned int iMu = 0; iMu < tcsMuonsVec2[iTC].size(); iMu++)
      if(tcsMuonsVec2[iTC][iMu].IsLive() ) {
        if(abs(16 - tcsMuonsVec2[iTC][iMu].GetEtaAddr()) <=7 )
          outputBarrelMuons.push_back(tcsMuonsVec2[iTC][iMu]);
        else
          outputEndcapMuons.push_back(tcsMuonsVec2[iTC][iMu]);
      }
      
  sort(outputBarrelMuons.begin(), outputBarrelMuons.end(), L1RpcTBMuon::TMuonMore());
  sort(outputEndcapMuons.begin(), outputEndcapMuons.end(), L1RpcTBMuon::TMuonMore());

  while(outputBarrelMuons.size() < L1RpcConst::FINAL_OUT_MUONS_CNT)
    outputBarrelMuons.push_back(L1RpcTBMuon());
  while(outputBarrelMuons.size() > L1RpcConst::FINAL_OUT_MUONS_CNT)
    outputBarrelMuons.pop_back();
  
  while(outputEndcapMuons.size() < L1RpcConst::FINAL_OUT_MUONS_CNT)
    outputEndcapMuons.push_back(L1RpcTBMuon());
  while(outputEndcapMuons.size() > L1RpcConst::FINAL_OUT_MUONS_CNT)
    outputEndcapMuons.pop_back();

  GBOutputMuons[0].insert(GBOutputMuons[0].end(), outputBarrelMuons.begin(), outputBarrelMuons.end());
  GBOutputMuons[1].insert(GBOutputMuons[1].end(), outputEndcapMuons.begin(), outputEndcapMuons.end());
  return GBOutputMuons;
}
/** 
 * Runs RunHalf() for 2 detecors parts.
 * Converts tower number (eta addr) from continous (0 - 32, tower 0 = 16)
 * to 2'complement.
 * @return 4 munons from barrel (GBOutputMuons[0]),
 * and 4 from endcaps (GBOutputMuons[1]).
*/
L1RpcTBMuonsVec2 L1RpcFinalSorter::Run(L1RpcTBMuonsVec2 &tcsMuonsVec2) {
  
  GBOutputMuons[0].clear();
  GBOutputMuons[1].clear();

  L1RpcTBMuonsVec2 firstHalfTcsMuonsVec2;

  firstHalfTcsMuonsVec2.push_back(tcsMuonsVec2[TrigCnfg->GetTCsCnt()-1]);
  for(int iTC = 0; iTC < TrigCnfg->GetTCsCnt()/2 +1; iTC++) {
    for(unsigned int iMu = 0; iMu < tcsMuonsVec2[iTC].size(); iMu++)
      tcsMuonsVec2[iTC][iMu].SetSectorAddr(iTC);
    firstHalfTcsMuonsVec2.push_back(tcsMuonsVec2[iTC]);
  }

  RunHalf(firstHalfTcsMuonsVec2);
  
  L1RpcTBMuonsVec2 secondHalfTcsMuonsVec2;
  for(int iTC = TrigCnfg->GetTCsCnt()/2-1; iTC < TrigCnfg->GetTCsCnt(); iTC++) {
    for(unsigned int iMu = 0; iMu < tcsMuonsVec2[iTC].size(); iMu++)
      tcsMuonsVec2[iTC][iMu].SetSectorAddr(iTC);
    secondHalfTcsMuonsVec2.push_back(tcsMuonsVec2[iTC]);
  }
  secondHalfTcsMuonsVec2.push_back(tcsMuonsVec2[0]);

  RunHalf(secondHalfTcsMuonsVec2);
  // Debug
  if (TrigCnfg->GetDebugLevel()!=0){
    for (unsigned  int iTC = 0; iTC < GBOutputMuons.size(); iTC++){
        for (unsigned  int iTB = 0; iTB < GBOutputMuons[iTC].size(); iTB++){
#ifndef _STAND_ALONE
            LogDebug("RPCHwDebug")<<"GB 3 "<< GBOutputMuons[iTC][iTB].printDebugInfo(TrigCnfg->GetDebugLevel());
#else
            std::cout <<"GB 3 "<< GBOutputMuons[iTC][iTB].printDebugInfo(TrigCnfg->GetDebugLevel()) << std::endl;
#endif 
        }
    }
  }

  RunFinalSorter(GBOutputMuons);

  if (TrigCnfg->GetDebugLevel()!=0){
    for (unsigned  int iTC = 0; iTC < GBOutputMuons.size(); iTC++){
        for (unsigned  int iTB = 0; iTB < GBOutputMuons[iTC].size(); iTB++){
#ifndef _STAND_ALONE
            LogDebug("RPCHwDebug")<<"GB 4 "<< GBOutputMuons[iTC][iTB].printDebugInfo(TrigCnfg->GetDebugLevel());
#else
            std::cout<<"GB 4 "<< GBOutputMuons[iTC][iTB].printDebugInfo(TrigCnfg->GetDebugLevel()) << std::endl;
#endif
        }
    }
  }

  return GBOutputMuons;
}
/*
 *
 *
 *
*/
void L1RpcFinalSorter::RunFinalSorter(L1RpcTBMuonsVec2 &finalMuons) {
//---------sorting-----------------------------------------
  sort(finalMuons[0].begin(), finalMuons[0].end(), L1RpcTBMuon::TMuonMore()) ;
  sort(finalMuons[1].begin(), finalMuons[1].end(), L1RpcTBMuon::TMuonMore()) ;

//-------setting size to GBETA_OUT_MUONS_CNT----------------
  while(finalMuons[0].size() > L1RpcConst::FINAL_OUT_MUONS_CNT) {
    finalMuons[0].pop_back();
    finalMuons[1].pop_back();
  }

//-----setting 2'complement EtaAddr
  for(unsigned int iMu = 0; iMu < GBOutputMuons[0].size(); iMu++) {
    // 10 oct 2006 - moved to L1RpcTCGhostBusterSorter::Run 
    //finalMuons[0][iMu].SetEtaAddr(TrigCnfg->TowNum2TowNum2Comp(finalMuons[0][iMu].GetEtaAddr()));
    //finalMuons[1][iMu].SetEtaAddr(TrigCnfg->TowNum2TowNum2Comp(finalMuons[1][iMu].GetEtaAddr()));
   
    finalMuons[0][iMu].SetPhiAddr(finalMuons[0][iMu].GetContinSegmAddr());
    finalMuons[1][iMu].SetPhiAddr(finalMuons[1][iMu].GetContinSegmAddr());   
  }  
}
