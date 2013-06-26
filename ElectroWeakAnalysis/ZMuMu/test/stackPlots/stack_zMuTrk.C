// stack macro
// Pasquale Noli

#include <sstream>
#include <string>
#include "stack_common.h"

void stack_zMuTrk() {

  /// cuts common....
  TCut kin_common("zMuTrkDau1Pt>20 && zMuTrkDau2Pt>20 && zMuTrkDau1TrkIso<3.0 && zMuTrkDau2TrkIso<3.0 && abs(zMuTrkDau1Eta)<2.1 &&  abs(zMuTrkDau2Eta)<2.1  && (zMuTrkDau1HLTBit==1) &&  abs(zMuTrkDau1dxyFromBS)<0.2 &&  abs(zMuTrkDau2dxyFromBS)<0.2");





 
  TCut dau2Loose("zMuTrkDau2TrkChi2<10  && (zMuTrkDau2TrkNofStripHits + zMuTrkDau2TrkNofPixelHits)>10 && zMuTrkDau2TrkNofPixelHits > 0");

  TCut dau1TightWP2_hltAlso("zMuTrkDau1GlobalMuonBit==1 && zMuTrkDau1Chi2<10  && (zMuTrkDau1NofStripHits + zMuTrkDau1NofPixelHits)>10  && zMuTrkDau1NofMuonHits>0 && zMuTrkDau1NofMuMatches>1   && zMuTrkDau1TrackerMuonBit==1 && zMuTrkDau1HLTBit==1");


 
  TCut massCut("zMuTrkMass>60 && zMuTrkMass<120 ");




  makePlots("zMuTrkMass", "", kin_common + dau1TightWP2_hltAlso +  dau2Loose, 5 , "zMuTrk" , 0.0001, 200, 0 ,200, true);

  makePlots("zMuTrkMass", "", kin_common , 5 , "zMuTrk" , 0.0001, 200, 0 ,200, true, true);



}


