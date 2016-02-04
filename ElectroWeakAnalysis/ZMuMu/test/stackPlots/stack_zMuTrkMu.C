// stack macro
// Pasquale Noli

#include <sstream>
#include <string>
#include "stack_common.h"

void stack_zMuTrkMu() {

  /// cuts common....
  TCut kin_common("zMuTrkMuDau1Pt> 20 && zMuTrkMuDau2Pt>20 && zMuTrkMuDau1Iso03SumPt<3.0 && zMuTrkMuDau2Iso03SumPt < 3.0 && abs(zMuTrkMuDau1Eta)<2.4 &&  abs(zMuTrkMuDau2Eta)<2.4  && (zMuTrkMuDau1HLTBit==1 || zMuTrkMuDau2HLTBit==1) &&  abs(zMuTrkMuDau1dxyFromBS)<0.2 &&  abs(zMuTrkMuDau2dxyFromBS)<0.2");





  TCut dau1Loose("zMuTrkMuDau1GlobalMuonBit==0 && zMuTrkMuDau1TrkChi2<10  && (zMuTrkMuDau1TrkNofStripHits + zMuTrkMuDau1TrkNofPixelHits)>10 && zMuTrkMuDau1TrkNofPixelHits > 0");
  TCut dau2Loose("zMuTrkMuDau2GlobalMuonBit==0 && zMuTrkMuDau2TrkChi2<10  && (zMuTrkMuDau2TrkNofStripHits + zMuTrkMuDau2TrkNofPixelHits)>10 && zMuTrkMuDau2TrkNofPixelHits > 0");

  TCut dau1TightWP2_hltAlso("zMuTrkMuDau1GlobalMuonBit==1 && zMuTrkMuDau1Chi2<10  && (zMuTrkMuDau1NofStripHits + zMuTrkMuDau1NofPixelHits)>10  && zMuTrkMuDau1NofMuonHits>0 && zMuTrkMuDau1NofMuMatches>1   && zMuTrkMuDau1TrackerMuonBit==1 && zMuTrkMuDau1HLTBit==1");
  TCut dau2TightWP2_hltAlso("zMuTrkMuDau2GlobalMuonBit==1 && zMuTrkMuDau2Chi2<10  && (zMuTrkMuDau2NofStripHits + zMuTrkMuDau2NofPixelHits)>10  && zMuTrkMuDau2NofMuonHits>0   &&  zMuTrkMuDau2NofMuMatches>1 && zMuTrkMuDau2TrackerMuonBit==1 && zMuTrkMuDau2HLTBit==1");

 
  TCut massCut("zMuTrkMuMass>60 && zMuTrkMuMass<120 ");




  makePlots("zMuTrkMuMass", "", kin_common + (( dau1TightWP2_hltAlso &&  dau2Loose ) || (dau2TightWP2_hltAlso +  dau1Loose)), 5, "zMuTrkMu",  0.0001, 200, 0 ,200, true, true);

  //  makePlots("zMuTrkMuMass", "", kin_common + dau1TightWP1_hltAlso , 5, "zMuTrkMu",  0.001, 200, 0 ,200, true);

    //makePlots("zMuTrkMuMass", "", kin_common, 5, "zMuTrkMu",  0.001, 200, 0 ,200, true);


}


