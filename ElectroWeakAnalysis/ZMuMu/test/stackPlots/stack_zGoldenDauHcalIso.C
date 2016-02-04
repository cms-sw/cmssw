// stack macro
// Pasquale Noli

#include <sstream>
#include <string>
#include "stack_common.h"

void stack_zGoldenDauHcalIso() {

  //TCut cut_zGolden("zGoldenMass>60 && zGoldenMass<120 && zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 && zGoldenDau1HcalIso< 3000.0 && zGoldenDau1HcalIso < 3000.0 && abs(zGoldenDau1Eta)<2.4 &&  abs(zGoldenDau2Eta)<2.4 && zGoldenDau1Chi2<10 && zGoldenDau2Chi2<10 && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>9 && zGoldenDau1NofPixelHits>0 && ( zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>9 &&  zGoldenDau2NofPixelHits>0 && zGoldenDau1NofMuonHits>0 && zGoldenDau2NofMuonHits>0 && zGoldenDau1NofMuMatches>1 && zGoldenDau2NofMuMatches>1   && zGoldenDau1TrackerMuonBit==1 && zGoldenDau2TrackerMuonBit==1 && (zGoldenDau1HLTBit==1 || zGoldenDau2HLTBit==1) ");

  TCut kin_common_woIso("zGoldenDau1Pt>20 && zGoldenDau2Pt>20 && abs(zGoldenDau1Eta)<2.4 &&  abs(zGoldenDau2Eta)<2.4  && (zGoldenDau1HLTBit==1 || zGoldenDau2HLTBit==1)  && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 ");



  makePlots("zGoldenDau1Iso03HadEt", "zGoldenDau2Iso03HadEt", massCut + kin_common_woIso + ( ( dau1Loose  && dau2TightWP1_hltAlso) || ( dau2Loose  && dau1TightWP1_hltAlso )) ,  1, "zGoldenDauHcalIso",  0.01, 50, 0 ,10, true);
  hs->GetXaxis()->SetTitle("#sum Et (GeV/c) (hcal)");
  string yTag = "events/(.2 GeV/c)"; // use the correct rebin
  hs->GetYaxis()->SetTitle(yTag.c_str());
  hs->GetYaxis()->SetTitleOffset(1.2);
  c1->SaveAs("zGoldenDauHcalIso.eps");
  c1->SaveAs("zGoldenDauHcalIso.gif");
  //c1->SaveAs("zGoldenDauHcalIso.pdf");
}
