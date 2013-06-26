#ifndef STACK_COMMON_H
#define STACK_COMMON_H

#include <iostream>
using namespace std;
#include "TChain.h"




void makeListOfCandidates() {



/// cuts common....
TCut kin_common("zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 && zGoldenDau1Iso03SumPt< 3.0 && zGoldenDau2Iso03SumPt < 3.0 && abs(zGoldenDau1Eta)<2.4 &&  abs(zGoldenDau2Eta)<2.4  && (zGoldenDau1HLTBit==1 || zGoldenDau2HLTBit==1)  && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 ");

TCut kin_common_musta("zMuStaDau1Pt> 20 && zMuStaDau2Pt>20 && zMuStaDau1Iso03SumPt< 3.0 && zMuStaDau2Iso03SumPt < 3.0 && abs(zMuStaDau1Eta)<2.4 &&  abs(zMuStaDau2Eta)<2.4  && (zMuStaDau1HLTBit==1 || zMuStaDau2HLTBit==1)  && abs(zMuStaDau1dxyFromBS)<0.2 && abs(zMuStaDau2dxyFromBS)<0.2 ");

TCut kin_common_mutrkMu("zMuTrkMuDau1Pt> 20 && zMuTrkMuDau2Pt>20 && zMuTrkMuDau1Iso03SumPt< 3.0 && zMuTrkMuDau2Iso03SumPt < 3.0 && abs(zMuTrkMuDau1Eta)<2.4 &&  abs(zMuTrkMuDau2Eta)<2.4  && (zMuTrkMuDau1HLTBit==1 || zMuTrkMuDau2HLTBit==1)  && abs(zMuTrkMuDau1dxyFromBS)<0.2 && abs(zMuTrkMuDau2dxyFromBS)<0.2 ");


TCut kin_common_notIso("zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 && (zGoldenDau1Iso03SumPt> 3.0 || zGoldenDau2Iso03SumPt>3.0) && abs(zGoldenDau1Eta)<2.4 &&  abs(zGoldenDau2Eta)<2.4  && (zGoldenDau1HLTBit==1 || zGoldenDau2HLTBit==1)  && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 ");



TCut dau1Loose(" (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10 ");
TCut dau2Loose(" (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10 ");



TCut dau1TightWP2("zGoldenDau1Chi2<10  && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10  && zGoldenDau1NofMuonHits>0   && zGoldenDau1TrackerMuonBit==1");
TCut dau2TightWP2("zGoldenDau2Chi2<10  && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10  && zGoldenDau2NofMuonHits>0   && zGoldenDau2TrackerMuonBit==1");

TCut dau1TightWP1_hltAlso("zGoldenDau1Chi2<10  && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10 && zGoldenDau1NofPixelHits>0 && zGoldenDau1NofMuonHits>0 &&  zGoldenDau1NofMuMatches>1  && zGoldenDau1TrackerMuonBit==1 && abs(zGoldenDau1Eta)<2.1 && zGoldenDau1HLTBit==1");
TCut dau2TightWP1_hltAlso("zGoldenDau2Chi2<10  && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10 && zGoldenDau2NofPixelHits>0 && zGoldenDau2NofMuonHits>0 &&  zGoldenDau2NofMuMatches>1  && zGoldenDau2TrackerMuonBit==1 && abs(zGoldenDau2Eta)<2.1 && zGoldenDau2HLTBit==1");


 
TCut massCut("zGoldenMass>60 && zGoldenMass<120 ");




TChain * dataEvents= new TChain("Events");


 dataEvents->Add("../NtupleLoose_132440_139790.root");
 dataEvents->Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_139_965_971.root");
 dataEvents->Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_139_972_980.root");
 dataEvents->Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_140_058_076.root");
 dataEvents->Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_140_116_126.root");






 std::cout << "

Z-->mu mu golden candidate in the mass range [60-120] " << std::endl;
Events->Scan("zGoldenRunNumber:zGoldenLumiblock:zGoldenEventNumber:zGoldenMass:zGoldenPt:zGoldenDau1Eta:zGoldenDau2Eta:zGoldenDau1Pt:zGoldenDau2Pt",  "zGoldenMass>60 && zGoldenMass<120" + kin_common + ( ( dau1Loose  && dau2TightWP1_hltAlso ) || ( dau2Loose  && dau1TightWP1_hltAlso )) );

 std::cout << "

run number Z-->mu mu golden candidate in the mass range [60-120] " << std::endl;
Events->Scan("zGoldenRunNumber",  "zGoldenMass>60 && zGoldenMass<120" + kin_common + ( ( dau1Loose  && dau2TightWP1_hltAlso ) || ( dau2Loose  && dau1TightWP1_hltAlso )) );


 std::cout << "

Z-->mu mu candidate outside the mass range [60-120] " << std::endl;
Events->Scan("zGoldenRunNumber:zGoldenLumiblock:zGoldenEventNumber:zGoldenMass:zGoldenPt:zGoldenDau1Eta:zGoldenDau2Eta:zGoldenDau1Pt:zGoldenDau2Pt:zGoldenDau1Iso03SumPt:zGoldenDau2Iso03SumPt:zGoldenDau1Iso03EmEt:zGoldenDau2Iso03EmEt:zGoldenDau1Iso03HadEt:zGoldenDau2Iso03HadEt:zGoldenDau1NofMuonHits:zGoldenDau2NofMuonHits:zGoldenDau1NofPixelHits:zGoldenDau2NofPixelHits",  "zGoldenMass<60 || zGoldenMass>120" + kin_common + ( ( dau1Loose  && dau2TightWP1_hltAlso ) || ( dau2Loose  && dau1TightWP1_hltAlso )) );

 std::cout << "

Z-->mu mu candidate not isolated in the mass range [60-120] " << std::endl;
Events->Scan("zGoldenRunNumber:zGoldenLumiblock:zGoldenEventNumber:zGoldenMass:zGoldenPt:zGoldenDau1Iso03SumPt:zGoldenDau2Iso03SumPt:zGoldenDau1Eta:zGoldenDau2Eta:zGoldenDau1Pt:zGoldenDau2Pt",  "zGoldenMass>60 && zGoldenMass<120" + kin_common_notIso + ( ( dau1Loose  && dau2TightWP1_hltAlso ) || ( dau2Loose  && dau1TightWP1_hltAlso )) );

  std::cout << "

Z-->mu sta  candidate in the mass range [60-120] " << std::endl;
Events->Scan("zMuStaRunNumber:zMuStaLumiblock:zMuStaEventNumber:zMuStaMass:zMuStaPt:zMuStaDau1Eta:zMuStaDau2Eta:zMuStaDau1Pt:zMuStaDau2Pt",  "zMuStaMass>60 && zMuStaMass<120" + kin_common_musta  );


std::cout << "

Z-->mu trk  candidate in the mass range [60-120] " << std::endl;
Events->Scan("zMuTrkRunNumber:zMuTrkLumiblock:zMuTrkEventNumber:zMuTrkMass:zMuTrkPt:zMuTrkDau1Eta:zMuTrkDau2Eta:zMuTrkDau1Pt:zMuTrkDau2Pt",  "zMuTrkMass>60 && zMuTrkMass<120" + kin_common_mutrkMu  );


}
           


