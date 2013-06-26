// stack macro
// Pasquale Noli

#include <sstream>
#include <string>
#include "stack_common.h"

void stack_zGolden() {


  //  TCut kin_common("zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 && zGoldenDau1TrkIso< 3.0 && zGoldenDau1TrkIso < 3.0 && abs(zGoldenDau1Eta)<2.4 &&  abs(zGoldenDau2Eta)<2.4  && (zGoldenDau1HLTBit==1 || zGoldenDau2HLTBit==1)  && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 ");



  // TCut dau1Loose(" (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10 ");
  // TCut dau2Loose(" (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10 ");

  //  TCut dau1TightWP1("zGoldenDau1Chi2<10  && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10 && zGoldenDau1NofPixelHits>0 && zGoldenDau1NofMuonHits>0 &&  zGoldenDau1NofMuMatches>1  && zGoldenDau1TrackerMuonBit==1");
  //TCut dau2TightWP1("zGoldenDau2Chi2<10  && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10 && zGoldenDau2NofPixelHits>0 && zGoldenDau2NofMuonHits>0 &&  zGoldenDau2NofMuMatches>1  && zGoldenDau2TrackerMuonBit==1");


  //TCut dau1TightWP2("zGoldenDau1Chi2<10  && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10  && zGoldenDau1NofMuonHits>0   && zGoldenDau1TrackerMuonBit==1");
  //  TCut dau2TightWP2("zGoldenDau2Chi2<10  && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10  && zGoldenDau2NofMuonHits>0   && zGoldenDau2TrackerMuonBit==1");

  // TCut dau1TightWP1_hltAlso("zGoldenDau1Chi2<10  && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10  && zGoldenDau1NofMuonHits>0   && zGoldenDau1TrackerMuonBit==1 && zGoldenDau1HLTBit==1");
  //  TCut dau2TightWP1_hltAlso("zGoldenDau2Chi2<10  && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10  && zGoldenDau2NofMuonHits>0   && zGoldenDau2TrackerMuonBit==1 && zGoldenDau2HLTBit==1");

 




    //TCut cut_zGolden("zGoldenDau1Chi2<10 && zGoldenDau2Chi2<10  && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10 && zGoldenDau1NofPixelHits>0 && (zGoldenDau2NofStripHits+ zGoldenDau2NofPixelHits)>10 &&  zGoldenDau2NofPixelHits>0 && zGoldenDau1NofMuonHits>0 && zGoldenDau2NofMuonHits>0 && zGoldenDau1NofMuMatches>1 && zGoldenDau2NofMuMatches>1 && zGoldenDau1TrackerMuonBit==1 && zGoldenDau2TrackerMuonBit==1");


  //  makePlots("zGoldenMass", "", kin_common + dau1TightWP1 + dau2TightWP1, 5, "zGolden",  0.001, 200, 0 ,200, true);




 // evaluateing now the total efficency of the qulaity cut

  //TCut massCut("zGoldenMass>60 && zGoldenMass<120 ");

  //evalEff("zGoldenMass", "",  kin_common + massCut, kin_common + massCut + dau1TightWP1  + dau2TightWP1  ,200, 0, 200); 



 //makePlots("zGoldenMass", "", kin_common + ( ( dau1Loose  && dau2TightWP1 ) || ( dau2Loose  && dau1TightWP1 )) , 5, "zGolden",  0.001, 200, 0 ,200, true);

 //evalEff("zGoldenMass", "",  kin_common + massCut, kin_common + massCut + ( ( dau1Loose  && dau2TightWP1 ) || ( dau2Loose  && dau1TightWP1 )) ,200, 0, 200); 

 // WP2

 //makePlots("zGoldenMass", "", kin_common + dau1TightWP2  + dau1TightWP2 , 5, "zGolden",  0.001, 200, 0 ,200, true);

 //evalEff("zGoldenMass", "",  kin_common + massCut, kin_common + massCut + dau2TightWP2 + dau1TightWP2,200, 0, 200); 

 // makePlots("zGoldenMass", "", kin_common + ( ( dau1Loose  && dau2TightWP2 ) || ( dau2Loose  && dau1TightWP2 )) , 5, "zGolden",  0.001, 200, 0 ,200, true);

 // evalEff("zGoldenMass", "",  kin_common + massCut, kin_common + massCut + ( ( dau1Loose  && dau2TightWP2 ) || ( dau2Loose  && dau1TightWP2 )) ,200, 0, 200); 

  makePlots("zGoldenMass", "",  kin_common + ( ( dau1Loose  && dau2TightWP1_hltAlso ) || ( dau2Loose  && dau1TightWP1_hltAlso )) ,5, "zGoldenLog_b5",  0.001, 200, 0 ,200, true, true);

 evalEff("zGoldenMass", "",  kin_common + massCut, kin_common + massCut + ( ( dau1Loose  && dau2TightWP1_hltAlso ) || ( dau2Loose  && dau1TightWP1_hltAlso )) ,200, 0, 200); 


}


