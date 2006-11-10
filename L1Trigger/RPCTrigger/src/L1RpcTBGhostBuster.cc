/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2004                                                      *
*                                                                              *
*******************************************************************************/

#include "L1Trigger/RPCTrigger/src/L1RpcTBGhostBuster.h"
//#include <set>
#include <algorithm>

using namespace std;
//---------------------------------------------------------------------------

L1RpcTBMuonsVec L1RpcTBGhostBuster::Run(L1RpcTBMuonsVec2 &pacMuonsVec2) const {
  L1RpcTBMuonsVec2 gbPhiMuonsVec2;
  for(unsigned int iTow = 0; iTow < pacMuonsVec2.size(); iTow++) {
    gbPhiMuonsVec2.push_back(GBPhi(pacMuonsVec2[iTow]) );
  }

  return GBEta(gbPhiMuonsVec2);
}


/* version wyit imlemented condition "last but one"
   NOT USED IN VHDL
  bool prevKillByBigger = false;
  for(int iMu = 0; iMu < SEGMENTS_IN_SECTOR_CNT; iMu++) {
    if(pacMuonsVec[iMu].GetCode() < pacMuonsVec[iPrevMu].GetCode() )
      prevKillByBigger = true;
    else if(pacMuonsVec[iMu].GetCode() > pacMuonsVec[iPrevMu].GetCode() )
      prevKillByBigger = false;  

    if(pacMuonsVec[iMu].GetCode() == 0);
    else if(pacMuonsVec[iMu].GetCode() > pacMuonsVec[iPrevMu].GetCode() &&    //local maximum
            pacMuonsVec[iMu].GetCode() > pacMuonsVec[iMu + 1].GetCode()   )
      ;
    else if(pacMuonsVec[iMu].GetCode() >= pacMuonsVec[iPrevMu].GetCode() &&   //last-but-one
            pacMuonsVec[iMu].GetCode() == pacMuonsVec[iMu + 1].GetCode() &&
            pacMuonsVec[iMu + 1].GetCode() > pacMuonsVec[iMu + 2].GetCode() ) {
      if(prevKillByBigger)
        pacMuonsVec[iMu].Kill();
    }
    else {
      pacMuonsVec[iMu].Kill();
    }

    iPrevMu = iMu;
  }
*/

L1RpcTBMuonsVec L1RpcTBGhostBuster::GBPhi(L1RpcTBMuonsVec &pacMuonsVec) const {
  if(pacMuonsVec.size() == 0)
    return L1RpcTBMuonsVec(); //empty vector;
//--------- killing ghosts ---------------------------------------
  pacMuonsVec.push_back(L1RpcTBMuon()); //adding  empty muon to the end,

  for(unsigned int iMu = 0; iMu < L1RpcConst::SEGMENTS_IN_SECTOR_CNT; iMu++) {
    if(pacMuonsVec[iMu].GetCode() == 0)
      ;
    else if(pacMuonsVec[iMu].GetCode() < pacMuonsVec[iMu + 1].GetCode() )
      pacMuonsVec[iMu].Kill();
    else if(pacMuonsVec[iMu].GetCode() == pacMuonsVec[iMu + 1].GetCode() ) {
      if(pacMuonsVec[iMu].WasKilled())
        pacMuonsVec[iMu+1].Kill();
      else
        pacMuonsVec[iMu].Kill();
    }
    else  //>
      pacMuonsVec[iMu+1].Kill();
  }

  pacMuonsVec.pop_back();  //removing empty muon from the end,

//-------setting the GBData ----------------------------------
  if(pacMuonsVec[0].IsLive() )
    pacMuonsVec[0].SetGBDataKilledFirst();
  else if(pacMuonsVec[0].WasKilled())
    for(unsigned int iMu = 0; iMu < L1RpcConst::SEGMENTS_IN_SECTOR_CNT; iMu++) {
      if(pacMuonsVec[iMu].IsLive() ) {
        pacMuonsVec[iMu].SetGBDataKilledFirst();
        break;
      }
    }

    if(pacMuonsVec[L1RpcConst::SEGMENTS_IN_SECTOR_CNT-1].IsLive() ) 
      pacMuonsVec[L1RpcConst::SEGMENTS_IN_SECTOR_CNT-1].SetGBDataKilledLast();
    else if(pacMuonsVec[L1RpcConst::SEGMENTS_IN_SECTOR_CNT-1].WasKilled())
      for(int iMu = L1RpcConst::SEGMENTS_IN_SECTOR_CNT -1; iMu >= 0 ; iMu--) {
        if(pacMuonsVec[iMu].IsLive() ) {
          pacMuonsVec[iMu].SetGBDataKilledLast();
          break;
      }
    }
//-------------sorting ------------------------------------------
  /*
  multiset<L1RpcTBMuon, L1RpcTBMuon::TMuonMore> liveMuonsSet;
  for(int iMu = 0; iMu < SEGMENTS_IN_SECTOR_CNT; iMu++) {
    if(pacMuonsVec[iMu].IsLive() ) {
      pacMuonsVec[iMu].SetPhiAddr(iMu);
      liveMuonsSet.insert(pacMuonsVec[iMu]);
    }
  }
  L1RpcTBMuonsVec outputMuons(liveMuonsSet.begin(), liveMuonsSet.end() );*/

  L1RpcTBMuonsVec outputMuons;
  for(unsigned int iMu = 0; iMu < L1RpcConst::SEGMENTS_IN_SECTOR_CNT; iMu++) {
    if(pacMuonsVec[iMu].IsLive() ) {
      pacMuonsVec[iMu].SetPhiAddr(iMu);
      outputMuons.push_back(pacMuonsVec[iMu]);
    }
  }
  sort(outputMuons.begin(), outputMuons.end(), L1RpcTBMuon::TMuonMore());

//-------setting size to GBPHI_OUT_MUONS_CNT----------------
  while (outputMuons.size() < L1RpcConst::GBPHI_OUT_MUONS_CNT)
    outputMuons.push_back(L1RpcTBMuon() );
  while(outputMuons.size() > L1RpcConst::GBPHI_OUT_MUONS_CNT)
    outputMuons.pop_back();

  return outputMuons;
}

////////////////////////////////////////////////////////////////////////////////

L1RpcTBMuonsVec L1RpcTBGhostBuster::GBEta(L1RpcTBMuonsVec2 &gbPhiMuonsVec2) const {
//-----  killing ghosts ---------------------------------------
  for(unsigned int iMuVec = 0; iMuVec < gbPhiMuonsVec2.size() -1; iMuVec++) {
    for(unsigned int iMu = 0; iMu < gbPhiMuonsVec2[iMuVec].size(); iMu++) {
      if(gbPhiMuonsVec2[iMuVec][iMu].GetCode() == 0)
        break; //because muons are sorted

      for(unsigned int iMuNext = 0; iMuNext < gbPhiMuonsVec2[iMuVec+1].size(); iMuNext++)
        if(abs(gbPhiMuonsVec2[iMuVec][iMu].GetPhiAddr() - gbPhiMuonsVec2[iMuVec+1][iMuNext].GetPhiAddr()) <= 1)
          if(gbPhiMuonsVec2[iMuVec][iMu].GetCode() < gbPhiMuonsVec2[iMuVec+1][iMuNext].GetCode())  //comparing with next
            gbPhiMuonsVec2[iMuVec][iMu].Kill();
          else
            gbPhiMuonsVec2[iMuVec+1][iMuNext].Kill();
    }
  }

//---------sorting-----------------------------------------
/*  multiset<L1RpcTBMuon, L1RpcTBMuon::TMuonMore> liveMuonsSet;
  for(unsigned int iMuVec = 0; iMuVec < gbPhiMuonsVec2.size(); iMuVec++)
  for(unsigned int iMu = 0; iMu < gbPhiMuonsVec2[iMuVec].size(); iMu++)
      if(gbPhiMuonsVec2[iMuVec][iMu].IsLive() ) {
        gbPhiMuonsVec2[iMuVec][iMu].SetEtaAddr(iMuVec);
        liveMuonsSet.insert(gbPhiMuonsVec2[iMuVec][iMu]);
      }
  L1RpcTBMuonsVec outputMuons(liveMuonsSet.begin(), liveMuonsSet.end() ); */

  L1RpcTBMuonsVec outputMuons;
  for(unsigned int iMuVec = 0; iMuVec < gbPhiMuonsVec2.size(); iMuVec++)
    for(unsigned int iMu = 0; iMu < gbPhiMuonsVec2[iMuVec].size(); iMu++)
      if(gbPhiMuonsVec2[iMuVec][iMu].IsLive() ) {
        gbPhiMuonsVec2[iMuVec][iMu].SetEtaAddr(iMuVec);
        outputMuons.push_back(gbPhiMuonsVec2[iMuVec][iMu]);
      }
  sort(outputMuons.begin(), outputMuons.end(), L1RpcTBMuon::TMuonMore());
  
//-------setting size to GBETA_OUT_MUONS_CNT----------------
  while(outputMuons.size() < L1RpcConst::GBETA_OUT_MUONS_CNT)
    outputMuons.push_back(L1RpcTBMuon());
  while(outputMuons.size() > L1RpcConst::GBETA_OUT_MUONS_CNT)
    outputMuons.pop_back();

  return outputMuons;
}


