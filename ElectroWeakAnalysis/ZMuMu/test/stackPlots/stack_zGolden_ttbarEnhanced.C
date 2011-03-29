// stack macro
// Pasquale Noli

#include <sstream>
#include <string>
#include "stack_common_ttbarEnhanced.h"


void stack_zGolden_ttbarEnhanced() {



  std::cout<< "combIso " << std::endl;

  //  makePlots("zGoldenMass", "", kin_common + dau1TightWP1_notChi2AndTrackerMuon  + dau2TightWP1_notChi2AndTrackerMuon + cut_1Iso + cut_2Iso + antiCosmicCut + "abs(zGoldenDau1Eta)<2.1 && abs(zGoldenDau2Eta)<2.1" + "numJets>0 && metPt>-1", "" ,1, "zGoldenLog_b1_0-200_njetm0",  0.1, 200, 0 ,200, true, true, true);
  //  makePlots("zGoldenMass", "", kin_common + dau1TightWP1_notChi2AndTrackerMuon  + dau2TightWP1_notChi2AndTrackerMuon + cut_1Iso + cut_2Iso + antiCosmicCut + "abs(zGoldenDau1Eta)<2.1 && abs(zGoldenDau2Eta)<2.1" + "numJets>1 && metPt>-1", "" ,1, "zGoldenLog_b1_0-200_njetm1",  0.1, 200, 0 ,200, true, true, true);

  // evalEff("zGoldenMass", "",  kin_common + massCut, kin_common + massCut + ( ( dau1Loose  && dau2TightWP1_hltAlso ) || ( dau2Loose  && dau1TightWP1_hltAlso )) ,200, 0, 200); 


  std::cout<< "trkIso " << std::endl;

  makePlots("numJets", "", "zGoldenMass>60 && zGoldenMass<120" +  kin_common + dau1TightWP1_notChi2AndTrackerMuon  + dau2TightWP1_notChi2AndTrackerMuon + cut_Trk1Iso + cut_Trk2Iso + antiCosmicCut + "abs(zGoldenDau1Eta)<2.1 && abs(zGoldenDau2Eta)<2.1"   , "" ,1, "numJets",  0.1, 10, -0.5 ,9.5, true, true, true);



}


