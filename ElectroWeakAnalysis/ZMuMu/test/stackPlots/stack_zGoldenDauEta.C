// stack macro
// Pasquale Noli

#include <sstream>
#include <string>
#include "stack_common.h"

void stack_zGoldenDauEta() {


  //TCut cut_zGolden("zGoldenMass>60 && zGoldenMass<120 && zGoldenDau1Pt> 0 && zGoldenDau2Pt>0 && zGoldenDau1TrkIso< 3.0 && zGoldenDau1TrkIso < 3.0 && abs(zGoldenDau1Eta)<2.4 &&  abs(zGoldenDau2Eta)<2.4 && zGoldenDau1Chi2<10 && zGoldenDau2Chi2<10 && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits )>9 && zGoldenDau1NofPixelHits>0 && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits) >9 &&  zGoldenDau2NofPixelHits>0 && zGoldenDau1NofMuonHits>0 && zGoldenDau2NofMuonHits>0 && zGoldenDau1NofMuMatches>1 && zGoldenDau2NofMuMatches>1   && zGoldenDau1TrackerMuonBit==1 && zGoldenDau2TrackerMuonBit==1 && (zGoldenDau1HLTBit==1 || zGoldenDau2HLTBit==1) ");
  



  makePlots("zGoldenDau1Eta", "zGoldenDau2Eta", massCut + kin_common + ( ( dau1Loose  && dau2TightWP1_hltAlso ) || ( dau2Loose  && dau1TightWP1_hltAlso )),  1, "zGoldenDauEta",  0.001, 18, -2.7 ,2.7, true, false);
  hs->GetXaxis()->SetTitle("#eta_{#mu}");
  string yTag = "events/(0.1)"; // use the correct rebin
  hs->GetYaxis()->SetTitle(yTag.c_str());
  c1->SaveAs("zGoldenDauEta.eps");
  c1->SaveAs("zGoldenDauEta.gif");
  //  c1->SaveAs("zGoldenDauPt.pdf");
}
