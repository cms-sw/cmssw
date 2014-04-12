/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2004                                                      *
*                                                                              *
*******************************************************************************/

#include "L1Trigger/RPCTrigger/interface/RPCTBGhostBuster.h"
//#include <set>
#include <algorithm>

using namespace std;
//---------------------------------------------------------------------------

L1RpcTBMuonsVec RPCTBGhostBuster::run(L1RpcTBMuonsVec2 &pacMuonsVec2) const {
  L1RpcTBMuonsVec2 gbPhiMuonsVec2;
  for(unsigned int iTow = 0; iTow < pacMuonsVec2.size(); iTow++) {
    gbPhiMuonsVec2.push_back(gBPhi(pacMuonsVec2[iTow]));
  }

  return gBEta(gbPhiMuonsVec2);
}


/* version wyit imlemented condition "last but one"
   NOT USED IN VHDL
  bool prevKillByBigger = false;
  for(int iMu = 0; iMu < m_SEGMENTS_IN_SECTOR_CNT; iMu++) {
    if(pacMuonsVec[iMu].getCode() < pacMuonsVec[iPrevMu].getCode())
      prevKillByBigger = true;
    else if(pacMuonsVec[iMu].getCode() > pacMuonsVec[iPrevMu].getCode())
      prevKillByBigger = false;  

    if(pacMuonsVec[iMu].getCode() == 0);
    else if(pacMuonsVec[iMu].getCode() > pacMuonsVec[iPrevMu].getCode() &&    //local maximum
            pacMuonsVec[iMu].getCode() > pacMuonsVec[iMu + 1].getCode()  )
      ;
    else if(pacMuonsVec[iMu].getCode() >= pacMuonsVec[iPrevMu].getCode() &&   //last-but-one
            pacMuonsVec[iMu].getCode() == pacMuonsVec[iMu + 1].getCode() &&
            pacMuonsVec[iMu + 1].getCode() > pacMuonsVec[iMu + 2].getCode()) {
      if(prevKillByBigger)
        pacMuonsVec[iMu].kill();
    }
    else {
      pacMuonsVec[iMu].kill();
    }

    iPrevMu = iMu;
  }
*/

L1RpcTBMuonsVec RPCTBGhostBuster::gBPhi(L1RpcTBMuonsVec &pacMuonsVec) const {
  if(pacMuonsVec.size() == 0)
    return L1RpcTBMuonsVec(); //empty vector;
//--------- killing ghosts ---------------------------------------
  pacMuonsVec.push_back(RPCTBMuon()); //adding  empty muon to the end,

  for(unsigned int iMu = 0; iMu < RPCConst::m_SEGMENTS_IN_SECTOR_CNT; iMu++) {
    if(pacMuonsVec[iMu].getCode() == 0)
      ;
    else if(pacMuonsVec[iMu].getCode() < pacMuonsVec[iMu + 1].getCode())
      pacMuonsVec[iMu].kill();
    else if(pacMuonsVec[iMu].getCode() == pacMuonsVec[iMu + 1].getCode()) {
      if(pacMuonsVec[iMu].wasKilled())
        pacMuonsVec[iMu+1].kill();
      else
        pacMuonsVec[iMu].kill();
    }
    else  //>
      pacMuonsVec[iMu+1].kill();
  }

  pacMuonsVec.pop_back();  //removing empty muon from the end,

//-------setting the m_GBData ----------------------------------
  if(pacMuonsVec[0].isLive())
    pacMuonsVec[0].setGBDataKilledFirst();
  else if(pacMuonsVec[0].wasKilled())
    for(unsigned int iMu = 0; iMu < RPCConst::m_SEGMENTS_IN_SECTOR_CNT; iMu++) {
      if(pacMuonsVec[iMu].isLive()) {
        pacMuonsVec[iMu].setGBDataKilledFirst();
        break;
      }
    }

    if(pacMuonsVec[RPCConst::m_SEGMENTS_IN_SECTOR_CNT-1].isLive()) 
      pacMuonsVec[RPCConst::m_SEGMENTS_IN_SECTOR_CNT-1].setGBDataKilledLast();
    else if(pacMuonsVec[RPCConst::m_SEGMENTS_IN_SECTOR_CNT-1].wasKilled())
      for(int iMu = RPCConst::m_SEGMENTS_IN_SECTOR_CNT -1; iMu >= 0 ; iMu--) {
        if(pacMuonsVec[iMu].isLive()) {
          pacMuonsVec[iMu].setGBDataKilledLast();
          break;
      }
    }
//-------------sorting ------------------------------------------
  /*
  multiset<RPCTBMuon, RPCTBMuon::TMuonMore> liveMuonsSet;
  for(int iMu = 0; iMu < m_SEGMENTS_IN_SECTOR_CNT; iMu++) {
    if(pacMuonsVec[iMu].isLive()) {
      pacMuonsVec[iMu].setPhiAddr(iMu);
      liveMuonsSet.insert(pacMuonsVec[iMu]);
    }
  }
  L1RpcTBMuonsVec outputMuons(liveMuonsSet.begin(), liveMuonsSet.end());*/

  L1RpcTBMuonsVec outputMuons;
  for(unsigned int iMu = 0; iMu < RPCConst::m_SEGMENTS_IN_SECTOR_CNT; iMu++) {
    if(pacMuonsVec[iMu].isLive()) {
      pacMuonsVec[iMu].setPhiAddr(iMu);
      outputMuons.push_back(pacMuonsVec[iMu]);
    }
  }
  sort(outputMuons.begin(), outputMuons.end(), RPCTBMuon::TMuonMore());

//-------setting size to m_GBPHI_OUT_MUONS_CNT----------------
  while (outputMuons.size() < RPCConst::m_GBPHI_OUT_MUONS_CNT)
    outputMuons.push_back(RPCTBMuon());
  while(outputMuons.size() > RPCConst::m_GBPHI_OUT_MUONS_CNT)
    outputMuons.pop_back();

  return outputMuons;
}

////////////////////////////////////////////////////////////////////////////////

L1RpcTBMuonsVec RPCTBGhostBuster::gBEta(L1RpcTBMuonsVec2 &gbPhiMuonsVec2) const {
//-----  killing ghosts ---------------------------------------
  for(unsigned int iMuVec = 0; iMuVec < gbPhiMuonsVec2.size() -1; iMuVec++) {
    for(unsigned int iMu = 0; iMu < gbPhiMuonsVec2[iMuVec].size(); iMu++) {
      if(gbPhiMuonsVec2[iMuVec][iMu].getCode() == 0)
        break; //because muons are sorted

      for(unsigned int iMuNext = 0; iMuNext < gbPhiMuonsVec2[iMuVec+1].size(); iMuNext++) {
        if(abs(gbPhiMuonsVec2[iMuVec][iMu].getPhiAddr()-gbPhiMuonsVec2[iMuVec+1][iMuNext].getPhiAddr())<=1)
        {
          //comparing with next:
          if(gbPhiMuonsVec2[iMuVec][iMu].getCode() < gbPhiMuonsVec2[iMuVec+1][iMuNext].getCode()) 
          { 
            gbPhiMuonsVec2[iMuVec][iMu].kill();
          }
          else 
          {
            gbPhiMuonsVec2[iMuVec+1][iMuNext].kill();
          }
        }
      }
    }
  }

//---------sorting-----------------------------------------
/*  multiset<RPCTBMuon, RPCTBMuon::TMuonMore> liveMuonsSet;
  for(unsigned int iMuVec = 0; iMuVec < gbPhiMuonsVec2.size(); iMuVec++)
  for(unsigned int iMu = 0; iMu < gbPhiMuonsVec2[iMuVec].size(); iMu++)
      if(gbPhiMuonsVec2[iMuVec][iMu].isLive()) {
        gbPhiMuonsVec2[iMuVec][iMu].setEtaAddr(iMuVec);
        liveMuonsSet.insert(gbPhiMuonsVec2[iMuVec][iMu]);
      }
  L1RpcTBMuonsVec outputMuons(liveMuonsSet.begin(), liveMuonsSet.end()); */

  L1RpcTBMuonsVec outputMuons;
  for(unsigned int iMuVec = 0; iMuVec < gbPhiMuonsVec2.size(); iMuVec++)
    for(unsigned int iMu = 0; iMu < gbPhiMuonsVec2[iMuVec].size(); iMu++)
      if(gbPhiMuonsVec2[iMuVec][iMu].isLive()) {
        gbPhiMuonsVec2[iMuVec][iMu].setEtaAddr(iMuVec);
        outputMuons.push_back(gbPhiMuonsVec2[iMuVec][iMu]);
      }
  sort(outputMuons.begin(), outputMuons.end(), RPCTBMuon::TMuonMore());
  
//-------setting size to m_GBETA_OUT_MUONS_CNT----------------
  while(outputMuons.size() < RPCConst::m_GBETA_OUT_MUONS_CNT)
    outputMuons.push_back(RPCTBMuon());
  while(outputMuons.size() > RPCConst::m_GBETA_OUT_MUONS_CNT)
    outputMuons.pop_back();

  return outputMuons;
}


