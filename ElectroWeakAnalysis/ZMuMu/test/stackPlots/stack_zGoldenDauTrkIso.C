// stack macro
// Pasquale Noli

#include <sstream>
#include <string>
#include "stack_common.h"

void stack_zGoldenDauTrkIso() {
TCut cut_zGolden("zGoldenMass>60 && zGoldenMass<120 && zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 && zGoldenDau1TrkIso< 3000.0 && zGoldenDau1TrkIso < 3000.0 && abs(zGoldenDau1Eta)<2.4 &&  abs(zGoldenDau2Eta)<2.4 && zGoldenDau1Chi2<10 && zGoldenDau2Chi2<10 && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2  && zGoldenDau1NofStripHits>9 && zGoldenDau1NofPixelHits>0 && zGoldenDau2NofStripHits>9 &&  zGoldenDau2NofPixelHits>0 && zGoldenDau1NofMuonHits>0 && zGoldenDau2NofMuonHits>0 && zGoldenDau1NofMuMatches>1 && zGoldenDau2NofMuMatches>1  && zGoldenDau1TrackerMuonBit==1 && zGoldenDau2TrackerMuonBit==1 && (zGoldenDau1HLTBit==1 || zGoldenDau2HLTBit==1) ");

 makePlots("zGoldenDau1TrkIso", "zGoldenDau2TrkIso",  cut_zGolden,  1, "zGoldenDauTrkIso",  0.01, 50, 0 ,10, true);
  hs->GetXaxis()->SetTitle("#sum pt (GeV/c) (tracker)");
  string yTag = "events/(.2 GeV/c)"; // use the correct rebin
  hs->GetYaxis()->SetTitle(yTag.c_str());
  c1->SaveAs("zGoldenDauTrkIso.eps");
  c1->SaveAs("zGoldenDauTrkIso.gif");
}
