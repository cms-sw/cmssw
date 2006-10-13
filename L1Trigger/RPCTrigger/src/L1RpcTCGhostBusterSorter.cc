//---------------------------------------------------------------------------

#include "L1Trigger/RPCTrigger/src/L1RpcTCGhostBusterSorter.h"

//#include <set>
#include <algorithm>
using namespace std;

//---------------------------------------------------------------------------
L1RpcTCGhostBusterSorter::L1RpcTCGhostBusterSorter(L1RpcTriggerConfiguration* triggerConfig) {
  TriggerConfig = triggerConfig;
}
//---------------------------------------------------------------------------
L1RpcTBMuonsVec L1RpcTCGhostBusterSorter::Run(L1RpcTBMuonsVec2 &tbMuonsVec2) {
  for (unsigned int iTB = 0; iTB < tbMuonsVec2.size()-1; iTB++) {
    for(unsigned int iMu = 0; iMu < tbMuonsVec2[iTB].size(); iMu++) {
      if(tbMuonsVec2[iTB][iMu].GetPtCode() == 0)
        break; //becouse muons are sorted
      if(tbMuonsVec2[iTB][iMu].GetEtaAddr() == (TriggerConfig->GetTowsCntOnTB(iTB)-1) ) {  //muon from this TB is on positive edge of TB (last tower of this tb)
        for(unsigned int iMuN = 0; iMuN < tbMuonsVec2[iTB + 1].size(); iMuN++) {
          if(tbMuonsVec2[iTB + 1][iMuN].GetPtCode() == 0)
            break; //becouse muons are sorted
          if(tbMuonsVec2[iTB+1][iMuN].GetEtaAddr() == 0) {  //muon from next TB is on negative edge (first tower of this TB)
            if( abs(tbMuonsVec2[iTB][iMu].GetPhiAddr() - tbMuonsVec2[iTB+1][iMuN].GetPhiAddr()) <= 1)
              if(tbMuonsVec2[iTB][iMu].GetCode() < tbMuonsVec2[iTB+1][iMuN].GetCode() )
                tbMuonsVec2[iTB][iMu].Kill();
              else
                tbMuonsVec2[iTB+1][iMuN].Kill();
          }
        }
      }
    }
  }

//---------sorting-----------------------------------------
/*  multiset<L1RpcTBMuon, L1RpcTBMuon::TMuonMore> liveMuonsSet;
  for(unsigned int iTB = 0; iTB < tbMuonsVec2.size(); iTB++)
  for(unsigned int iMu = 0; iMu < tbMuonsVec2[iTB].size(); iMu++)
      if(tbMuonsVec2[iTB][iMu].IsLive() ) {

        int etaAddr = tbMuonsVec2[iTB][iMu].GetEtaAddr() | (iTB<<2); //tower number natural
        etaAddr = TriggerConfig->TowAddr2TowNum(etaAddr); //tower number: -16 : 16
        etaAddr = etaAddr + 16;                     // tower number continous 0 : 32
        tbMuonsVec2[iTB][iMu].SetEtaAddr(etaAddr);

        liveMuonsSet.insert(tbMuonsVec2[iTB][iMu]);
      }
  L1RpcTBMuonsVec outputMuons(liveMuonsSet.begin(), liveMuonsSet.end() ); */

  L1RpcTBMuonsVec outputMuons;
  for(unsigned int iTB = 0; iTB < tbMuonsVec2.size(); iTB++)
    for(unsigned int iMu = 0; iMu < tbMuonsVec2[iTB].size(); iMu++)
      if(tbMuonsVec2[iTB][iMu].IsLive() ) {
        int etaAddr = tbMuonsVec2[iTB][iMu].GetEtaAddr() | (iTB<<2); //tower number natural <0...35>
        etaAddr = TriggerConfig->TowAddr2TowNum(etaAddr); //tower number: -16 : 16
        etaAddr = etaAddr + 16;                     // tower number continous 0 : 32
        etaAddr = TriggerConfig->TowNum2TowNum2Comp(etaAddr); // 10 oct 2006 - moved from FS
        tbMuonsVec2[iTB][iMu].SetEtaAddr(etaAddr);

        outputMuons.push_back(tbMuonsVec2[iTB][iMu]);
      }

  sort(outputMuons.begin(), outputMuons.end(), L1RpcTBMuon::TMuonMore());
  
//-------setting size to GBETA_OUT_MUONS_CNT----------------
  while(outputMuons.size() < L1RpcConst::TCGB_OUT_MUONS_CNT)
    outputMuons.push_back(L1RpcTBMuon());
  while(outputMuons.size() > L1RpcConst::TCGB_OUT_MUONS_CNT)
    outputMuons.pop_back();

  return outputMuons;
}
