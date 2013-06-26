#ifndef STACK_COMMON_H
#define STACK_COMMON_H

#include <iostream>
using namespace std;
#include "TChain.h"




void qualityCutCheck() {



/// cuts common....
TCut kin_common("zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 && zGoldenDau1Iso03SumPt< 3.0 && zGoldenDau2Iso03SumPt < 3.0 && abs(zGoldenDau1Eta)<2.4 &&  abs(zGoldenDau2Eta)<2.4  && (zGoldenDau1HLTBit==1 || zGoldenDau2HLTBit==1)  && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 ");

TCut kin_common_musta("zMuStaDau1Pt> 20 && zMuStaDau2Pt>20 && zMuStaDau1Iso03SumPt< 3.0 && zMuStaDau2Iso03SumPt < 3.0 && abs(zMuStaDau1Eta)<2.4 &&  abs(zMuStaDau2Eta)<2.4  && (zMuStaDau1HLTBit==1 || zMuStaDau2HLTBit==1)  && abs(zMuStaDau1dxyFromBS)<0.2 && abs(zMuStaDau2dxyFromBS)<0.2 ");

TCut kin_common_notIso("zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 && (zGoldenDau1Iso03SumPt> 3.0 || zGoldenDau2Iso03SumPt>3.0) && abs(zGoldenDau1Eta)<2.4 &&  abs(zGoldenDau2Eta)<2.4  && (zGoldenDau1HLTBit==1 || zGoldenDau2HLTBit==1)  && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 ");



TCut dau1Loose(" (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10 ");
TCut dau2Loose(" (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10 ");

TCut dau1TightWP1("zGoldenDau1Chi2<10  && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10 && zGoldenDau1NofPixelHits>0 && zGoldenDau1NofMuonHits>0 &&  zGoldenDau1NofMuMatches>1  && zGoldenDau1TrackerMuonBit==1");
TCut dau2TightWP1("zGoldenDau2Chi2<10  && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10 && zGoldenDau2NofPixelHits>0 && zGoldenDau2NofMuonHits>0 &&  zGoldenDau2NofMuMatches>1  && zGoldenDau2TrackerMuonBit==1");


TCut dau1TightWP2("zGoldenDau1Chi2<10  && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10  && zGoldenDau1NofMuonHits>0   && zGoldenDau1TrackerMuonBit==1");
TCut dau2TightWP2("zGoldenDau2Chi2<10  && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10  && zGoldenDau2NofMuonHits>0   && zGoldenDau2TrackerMuonBit==1");

TCut dau1TightWP1_hltAlso("zGoldenDau1Chi2<10  && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10 && zGoldenDau1NofPixelHits>0 && zGoldenDau1NofMuonHits>0 &&  zGoldenDau1NofMuMatches>1  && zGoldenDau1TrackerMuonBit==1 && zGoldenDau1HLTBit==1 && (abs(zGoldenDau1Eta)<2.1)");
TCut dau2TightWP1_hltAlso("zGoldenDau2Chi2<10  && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10 && zGoldenDau2NofPixelHits>0 && zGoldenDau2NofMuonHits>0 &&  zGoldenDau2NofMuMatches>1  && zGoldenDau2TrackerMuonBit==1&& zGoldenDau2HLTBit==0 && (abs(zGoldenDau2Eta)<2.1)");



 
TCut massCut("zGoldenMass>60 && zGoldenMass<120 ");




TChain * dataEvents= new TChain("Events");

 dataEvents->Add("/scratch2/users/degruttola/data/OfficialJSON/NtupleLoose_132440_139790.root");
 dataEvents->Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_139_965_971.root");
 dataEvents->Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_139_972_980.root");
 dataEvents->Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_140_058_076.root");
 dataEvents->Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_140_116_126.root");





 std::cout << "muon with high chi2 " << std::endl;

 Events->Scan("zGoldenRunNumber:zGoldenLumiblock:zGoldenEventNumber:zGoldenMass:zGoldenDau1Pt:zGoldenDau1Eta:zGoldenDau1Iso03SumPt:zGoldenDau1Chi2:zGoldenDau1NofPixelHits:zGoldenDau1NofStripHits:zGoldenDau1NofMuonHits:zGoldenDau1HLTBit:zGoldenDau1TrackerMuonBit:zGoldenDau1NofMuMatches:zGoldenDau1TrkChi2:zGoldenDau1SaChi2",  "zGoldenMass>60 && zGoldenMass<120" + kin_common + "zGoldenDau1Chi2>10 || zGoldenDau2Chi2>10");

 std::cout << "muon with hit  pixel 0 " << std::endl;


  Events->Scan("zGoldenRunNumber:zGoldenLumiblock:zGoldenEventNumber:zGoldenMass:zGoldenDau1Pt:zGoldenDau1Eta:zGoldenDau1Iso03SumPt:zGoldenDau1Chi2:zGoldenDau1NofPixelHits:zGoldenDau1NofStripHits:zGoldenDau1NofMuonHits:zGoldenDau1HLTBit:zGoldenDau1TrackerMuonBit:zGoldenDau1NofMuMatches:zGoldenDau1dxyFromBS:zGoldenDau2dxyFromBS",  "zGoldenMass>60 && zGoldenMass<120" + kin_common + "zGoldenDau1NofPixelHits==0 || zGoldenDau2NofPixelHits==0");

 std::cout << "muon chamber <2  " << std::endl;

Events->Scan("zGoldenRunNumber:zGoldenLumiblock:zGoldenEventNumber:zGoldenMass:zGoldenDau1Pt:zGoldenDau1Eta:zGoldenDau1Iso03SumPt:zGoldenDau1Chi2:zGoldenDau1NofPixelHits:zGoldenDau1NofStripHits:zGoldenDau1NofMuonHits:zGoldenDau1HLTBit:zGoldenDau1TrackerMuonBit:zGoldenDau1NofMuMatches:zGoldenDau1dxyFromBS:zGoldenDau2dxyFromBS",  "zGoldenMass>60 && zGoldenMass<120" + kin_common + "zGoldenDau1NofMuMatches<2 || zGoldenDau2NofMuMatches<2");

  //Events->Scan("zMuStaRunNumber:zMuStaLumiblock:zMuStaEventNumber:zMuStaMass:zMuStaDau1Pt:zMuStaDau1Eta:zMuStaDau2Pt:zMuStaDau2Eta:zMuStaDau1HLTBit:zMuStaDau1NofMuMatches:zMuStaDau2HLTBit:zMuStaDau2NofMuMatches:",  "zMuStaMass>60 && zMuStaMass<120"+ kin_common_musta );

 std::cout << "muon hit 0  " << std::endl;

Events->Scan("zGoldenRunNumber:zGoldenLumiblock:zGoldenEventNumber:zGoldenMass:zGoldenDau1Pt:zGoldenDau1Eta:zGoldenDau1Iso03SumPt:zGoldenDau1Chi2:zGoldenDau1NofPixelHits:zGoldenDau1NofStripHits:zGoldenDau1NofMuonHits:zGoldenDau1HLTBit:zGoldenDau1TrackerMuonBit:zGoldenDau1NofMuMatches:zGoldenDau1dxyFromBS:zGoldenDau2dxyFromBS",  "zGoldenMass>60 && zGoldenMass<120" + kin_common + "zGoldenDau1NofMuonHits==0 || zGoldenDau2NofMuonHits==0");



}
           


