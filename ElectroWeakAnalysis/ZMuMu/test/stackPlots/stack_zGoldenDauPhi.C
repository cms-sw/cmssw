// stack macro
// Pasquale Noli

#include <sstream>
#include <string>
#include "stack_common.h"

void stack_zGoldenDauPhi() {


  //TCut cut_zGolden("zGoldenMass>60 && zGoldenMass<120 && zGoldenDau1Pt> 0 && zGoldenDau2Pt>0 && zGoldenDau1TrkIso< 3.0 && zGoldenDau1TrkIso < 3.0 && abs(zGoldenDau1Eta)<2.4 &&  abs(zGoldenDau2Eta)<2.4 && zGoldenDau1Chi2<10 && zGoldenDau2Chi2<10 && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits )>9 && zGoldenDau1NofPixelHits>0 && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits) >9 &&  zGoldenDau2NofPixelHits>0 && zGoldenDau1NofMuonHits>0 && zGoldenDau2NofMuonHits>0 && zGoldenDau1NofMuMatches>1 && zGoldenDau2NofMuMatches>1   && zGoldenDau1TrackerMuonBit==1 && zGoldenDau2TrackerMuonBit==1 && (zGoldenDau1HLTBit==1 || zGoldenDau2HLTBit==1) ");
  



  makePlots("zGoldenDau1Phi", "zGoldenDau2Phi", massCut + kin_common + ( ( dau1Loose  && dau2TightWP1_hltAlso ) || ( dau2Loose  && dau1TightWP1_hltAlso )),  1, "zGoldenDauPhi",  0.001, 20,  -3.1415 , +3.1415 , true, false);
  hs->GetXaxis()->SetTitle("#phi_{#mu}");
  string yTag = "events/(20/6.28)"; // use the correct rebin
  hs->GetYaxis()->SetTitle(yTag.c_str());
  c1->SaveAs("zGoldenDauPhi.eps");
  c1->SaveAs("zGoldenDauPhi.gif");
  //  c1->SaveAs("zGoldenDauPt.pdf");
}
