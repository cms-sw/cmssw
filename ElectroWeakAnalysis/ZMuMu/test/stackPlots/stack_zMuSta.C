// stack macro
// Pasquale Noli

#include <sstream>
#include <string>
#include "stack_common.h"

void stack_zMuSta() {

  /// cuts common....
  TCut kin_common("zMuStaDau1Pt> 20 && zMuStaDau2Pt>20 && zMuStaDau1Iso03SumPt<3.0 && zMuStaDau2Iso03SumPt < 3.0 && abs(zMuStaDau1Eta)<2.1 &&  abs(zMuStaDau2Eta)<2.1  && (zMuStaDau1HLTBit==1 || zMuStaDau2HLTBit==1) && ( ( zMuStaDau1GlobalMuonBit==1 && abs(zMuStaDau1dxyFromBS)<0.2 ) || ( zMuStaDau2GlobalMuonBit==1 && abs(zMuStaDau2dxyFromBS)<0.2 ))");





  TCut dau1Loose("zMuStaDau1GlobalMuonBit==0 && zMuStaDau1SaChi2<10  && zMuStaDau1SaNofMuonHits>0 &&  zMuStaDau1NofMuMatches>1");
  TCut dau2Loose("zMuStaDau1GlobalMuonBit==0 && zMuStaDau2SaChi2<10  && zMuStaDau2SaNofMuonHits>0 &&  zMuStaDau2NofMuMatches>1");

  TCut dau1TightWP2_hltAlso("zMuStaDau1GlobalMuonBit==1 && zMuStaDau1Chi2<10  && (zMuStaDau1NofStripHits + zMuStaDau1NofPixelHits)>10  && zMuStaDau1NofMuonHits>0 && zMuStaDau1NofMuMatches>1   && zMuStaDau1TrackerMuonBit==1 && zMuStaDau1HLTBit==1");
  TCut dau2TightWP2_hltAlso("zMuStaDau2GlobalMuonBit==1 && zMuStaDau2Chi2<10  && (zMuStaDau2NofStripHits + zMuStaDau2NofPixelHits)>10  && zMuStaDau2NofMuonHits>0   &&  zMuStaDau2NofMuMatches>1 && zMuStaDau2TrackerMuonBit==1 && zMuStaDau2HLTBit==1");

 
  TCut massCut("zMuStaMass>60 && zMuStaMass<120 ");




  makePlots("zMuStaMass", "", kin_common + (( dau1TightWP2_hltAlso &&  dau2Loose ) || (dau2TightWP2_hltAlso +  dau1Loose)), 5, "zMuSta",  0.0001, 200, 0 ,200, true, true);

  //  makePlots("zMuStaMass", "", kin_common + dau1TightWP1_hltAlso , 5, "zMuSta",  0.001, 200, 0 ,200, true);

    //makePlots("zMuStaMass", "", kin_common, 5, "zMuSta",  0.001, 200, 0 ,200, true);


}


