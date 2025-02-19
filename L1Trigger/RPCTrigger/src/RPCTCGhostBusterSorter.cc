#include "L1Trigger/RPCTrigger/interface/RPCTCGhostBusterSorter.h"
//#include <set>
#include <algorithm>


using namespace std;

//---------------------------------------------------------------------------
RPCTCGhostBusterSorter::RPCTCGhostBusterSorter(RPCTriggerConfiguration* triggerConfig) {
  m_TriggerConfig = triggerConfig;
}
//---------------------------------------------------------------------------
L1RpcTBMuonsVec RPCTCGhostBusterSorter::run(L1RpcTBMuonsVec2 &tbMuonsVec2) {
  for (unsigned int iTB = 0; iTB < tbMuonsVec2.size()-1; iTB++) {
    for(unsigned int iMu = 0; iMu < tbMuonsVec2[iTB].size(); iMu++) {
      if(tbMuonsVec2[iTB][iMu].getPtCode() == 0)
        break; //becouse muons are sorted
      
      //muon from this TB is on positive edge of TB (last m_tower of this tb):
      if(tbMuonsVec2[iTB][iMu].getEtaAddr() == (m_TriggerConfig->getTowsCntOnTB(iTB)-1)) {  
        for(unsigned int iMuN = 0; iMuN < tbMuonsVec2[iTB + 1].size(); iMuN++) {
          if(tbMuonsVec2[iTB + 1][iMuN].getPtCode() == 0)
            break; //becouse muons are sorted
          
          //muon from next TB is on negative edge (first m_tower of this TB):
          if(tbMuonsVec2[iTB+1][iMuN].getEtaAddr() == 0) {  
            if( abs(tbMuonsVec2[iTB][iMu].getPhiAddr() - tbMuonsVec2[iTB+1][iMuN].getPhiAddr()) <= 1)
            {
              if(tbMuonsVec2[iTB][iMu].getCode() < tbMuonsVec2[iTB+1][iMuN].getCode())
              {
                tbMuonsVec2[iTB][iMu].kill();
              }
              else
              {
                tbMuonsVec2[iTB+1][iMuN].kill();
              }
            }
          }
        }
      }
    }
  }

//---------sorting-----------------------------------------
/*  multiset<RPCTBMuon, RPCTBMuon::TMuonMore> liveMuonsSet;
  for(unsigned int iTB = 0; iTB < tbMuonsVec2.size(); iTB++)
  for(unsigned int iMu = 0; iMu < tbMuonsVec2[iTB].size(); iMu++)
      if(tbMuonsVec2[iTB][iMu].isLive()) {

        int etaAddr = tbMuonsVec2[iTB][iMu].getEtaAddr() | (iTB<<2); //m_tower number natural
        etaAddr = m_TriggerConfig->towAddr2TowNum(etaAddr); //m_tower number: -16 : 16
        etaAddr = etaAddr + 16;                     // m_tower number continous 0 : 32
        tbMuonsVec2[iTB][iMu].setEtaAddr(etaAddr);

        liveMuonsSet.insert(tbMuonsVec2[iTB][iMu]);
      }
  L1RpcTBMuonsVec outputMuons(liveMuonsSet.begin(), liveMuonsSet.end()); */

  L1RpcTBMuonsVec outputMuons;
  for(unsigned int iTB = 0; iTB < tbMuonsVec2.size(); iTB++)
    for(unsigned int iMu = 0; iMu < tbMuonsVec2[iTB].size(); iMu++)
      if(tbMuonsVec2[iTB][iMu].isLive()) {
        int etaAddr = tbMuonsVec2[iTB][iMu].getEtaAddr() | (iTB<<2); //m_tower number natural <0...35>
        etaAddr = m_TriggerConfig->towAddr2TowNum(etaAddr); //m_tower number: -16 : 16
        //etaAddr = etaAddr + 16;                     // m_tower number continous 0 : 32
        etaAddr = m_TriggerConfig->towNum2TowNum2Comp(etaAddr); // 10 oct 2006 - moved from FS
        tbMuonsVec2[iTB][iMu].setEtaAddr(etaAddr);

        outputMuons.push_back(tbMuonsVec2[iTB][iMu]);
      }

  sort(outputMuons.begin(), outputMuons.end(), RPCTBMuon::TMuonMore());
  
//-------setting size to m_GBETA_OUT_MUONS_CNT----------------
  while(outputMuons.size() < RPCConst::m_TCGB_OUT_MUONS_CNT)
    outputMuons.push_back(RPCTBMuon());
  while(outputMuons.size() > RPCConst::m_TCGB_OUT_MUONS_CNT)
    outputMuons.pop_back();

  return outputMuons;
}
