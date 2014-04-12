#include "TFile.h"
#include "TH1F.h"
#include "TTree.h"
#include "TAxis.h"
#include "TCanvas.h"
//#if !defined(__CINT__) && !defined(__MAKECINT__)                                                                                                          
#include <string>
#include <iostream>
//#endif   

void setGraphics(TH1F *histo){

  histo->SetFillColor(kAzure+7);
  //histo->SetLineWidth(2);
  histo->SetLineColor(kBlue+1);
}




void plotsAfterCuts_BothLegs(){

  gStyle->SetOptStat();
  gROOT->SetStyle("Plain");
  using namespace std;


  TChain Events("Events"); 
  
  //     Events.Add("../../NtupleLoose_135_all.root")
  Events.Add("/scratch2/users/degruttola/data/OfficialJSON/blessed/NtupleLoose_132440_139790.root");
  Events.Add("/scratch2/users/degruttola/data/OfficialJSON/blessed/NtupleLoose_139791-140159_v2.root");
  Events.Add("/scratch2/users/degruttola/data/OfficialJSON/blessed/NtupleLoose_140160-140182.root");
  Events.Add("/scratch2/users/degruttola/data/OfficialJSON/blessed/NtupleLoose_140183-140399.root");
  Events.Add("/scratch2/users/degruttola/data/OfficialJSON/blessed/NtupleLoose_140440-141961.root");
  Events.Add("/scratch2/users/degruttola/data/OfficialJSON/blessed/NtupleLoose_142035-142664.root");
  Events.Add("/scratch2/users/degruttola/data/OfficialJSON/blessed//NtupleLoose_142665-143179.root");
  Events.Add("/scratch2/users/degruttola/data/OfficialJSON/blessed/NtupleLoose_143180-143336.root");
  Events.Add("/scratch2/users/degruttola/data/OfficialJSON/blessed/NtupleLoose_143337-144114.root");

  
  TFile * output_file = TFile::Open("histoZmm_2p88_afterCutsBothLegs.root", "RECREATE");


  TCut cut_1Iso("((zGoldenDau1Iso03SumPt + zGoldenDau1Iso03EmEt + zGoldenDau1Iso03HadEt)/ zGoldenDau1Pt ) < 0.15");
  TCut cutTk_1Iso("((zMuTrkDau1Iso03SumPt + zMuTrkDau1Iso03EmEt + zMuTrkDau1Iso03HadEt)/ zMuTrkDau1Pt ) < 0.15");
  TCut cutSa_1Iso("((zMuStaDau1Iso03SumPt + zMuStaDau1Iso03EmEt + zMuStaDau1Iso03HadEt)/ zMuStaDau1Pt ) < 0.15");

  TCut cut_2Iso("((zGoldenDau2Iso03SumPt + zGoldenDau2Iso03EmEt + zGoldenDau2Iso03HadEt)/ zGoldenDau2Pt ) < 0.15");
  TCut cutTk_2Iso("((zMuTrkDau2Iso03SumPt + zMuTrkDau2Iso03EmEt + zMuTrkDau2Iso03HadEt)/ zMuTrkDau2Pt ) < 0.15");
  TCut cutSa_2Iso("((zMuStaDau2Iso03SumPt + zMuStaDau2Iso03EmEt + zMuStaDau2Iso03HadEt)/ zMuStaDau2Pt ) < 0.15");




  TCut cut_OneNotIso("((((zGoldenDau1Iso03SumPt + zGoldenDau1Iso03EmEt + zGoldenDau1Iso03HadEt)/ zGoldenDau1Pt ) > 0.15) && (((zGoldenDau2Iso03SumPt + zGoldenDau2Iso03EmEt + zGoldenDau2Iso03HadEt)/ zGoldenDau2Pt ) < 0.15)) || ((((zGoldenDau1Iso03SumPt + zGoldenDau1Iso03EmEt + zGoldenDau1Iso03HadEt)/ zGoldenDau1Pt ) < 0.15) && (((zGoldenDau2Iso03SumPt + zGoldenDau2Iso03EmEt + zGoldenDau2Iso03HadEt)/ zGoldenDau2Pt ) > 0.15))"); 

TCut cut_TwoNotIso("(((zGoldenDau1Iso03SumPt + zGoldenDau1Iso03EmEt + zGoldenDau1Iso03HadEt)/ zGoldenDau1Pt ) > 0.15) && ((zGoldenDau2Iso03SumPt + zGoldenDau2Iso03EmEt + zGoldenDau2Iso03HadEt)/ zGoldenDau2Pt ) > 0.15");

TCut cut_NotIso("(((zGoldenDau1Iso03SumPt + zGoldenDau1Iso03EmEt + zGoldenDau1Iso03HadEt)/ zGoldenDau1Pt ) > 0.15) || ((zGoldenDau2Iso03SumPt + zGoldenDau2Iso03EmEt + zGoldenDau2Iso03HadEt)/ zGoldenDau2Pt ) > 0.15");


/// cuts common....
 TCut kin_common(" (zGoldenDau1Q * zGoldenDau2Q) ==-1 &&  zGoldenDau1Pt>20 && zGoldenDau2Pt>20 && abs(zGoldenDau1Eta)<2.1 && abs(zGoldenDau2Eta)<2.1  && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2  && ( zGoldenDau1HLTBit==1 ||  zGoldenDau2HLTBit==1) "); // 


TCut dau1TightWP1_notChi2AndTrackerMuon("zGoldenDau1Chi2<10000  && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10 && zGoldenDau1NofPixelHits>0 && zGoldenDau1NofMuonHits>0 &&  zGoldenDau1NofMuMatches>1");

TCut dau2TightWP1_notChi2AndTrackerMuon("zGoldenDau2Chi2<10000  && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10 && zGoldenDau2NofPixelHits>0 && zGoldenDau2NofMuonHits>0 &&  zGoldenDau2NofMuMatches>1");



  
  // zGolden plots
//  TCut cut_zGolden("zGoldenMass>60 && zGoldenMass<120 && zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 && abs(zGoldenDau1Eta)<2.1 &&  abs(zGoldenDau2Eta)<2.1 && zGoldenDau1Chi2<10000 && zGoldenDau2Chi2<10000 && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>=10 && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>=10 && zGoldenDau1NofMuonHits>0 && zGoldenDau2NofMuonHits>0 && (zGoldenDau1HLTBit==1 || zGoldenDau2HLTBit==1) ");
  TDirectory * dir = output_file->mkdir("goodZToMuMuPlots");
  dir->cd();
  TH1F * zMass = new TH1F("zMass", "zMass", 200, 0, 200);
  //  Events.Draw("zGoldenMass");
  Events.Project("zMass", "zGoldenMass", kin_common + dau1TightWP1_notChi2AndTrackerMuon  + dau2TightWP1_notChi2AndTrackerMuon +  cut_1Iso + cut_2Iso);
  cout<<"Number of zGoldenAA : "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;

  
  //TCut cut2_zGolden1HLT("zGoldenMass>60 && zGoldenMass<120 && zGoldenDau1Pt> 20 && zGoldenDau2Pt>20  && abs(zGoldenDau1Eta)<2.1 &&  abs(zGoldenDau2Eta)<2.1 && zGoldenDau1Chi2<1000 && zGoldenDau2Chi2<1000 && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 &&(zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>=10 && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>=10 && zGoldenDau1NofMuonHits>0 && zGoldenDau2NofMuonHits>0 && ( zGoldenDau1HLTBit ==0 || zGoldenDau2HLTBit ==0)");
  TDirectory * dir = output_file->mkdir("goodZToMuMu1HLTPlots");
  dir->cd();
  TH1F * zMass = new TH1F("zMass", "zMass", 200, 0, 200);
    Events.Project("zMass", "zGoldenMass", kin_common + dau1TightWP1_notChi2AndTrackerMuon  + dau2TightWP1_notChi2AndTrackerMuon +  cut_1Iso + cut_2Iso + "zGoldenDau1HLTBit==0 || zGoldenDau2HLTBit==0" );
  cout<<"Number of zGolden1HLT : "<<zMass->GetEntries()<<endl;
 zMass->Write();
  delete zMass;
 //zMass2->Write();

  output_file->cd("/");


  
  

  output_file->cd("/");

  //   TCut cut2_zGolden2HLT("zGoldenMass>60 && zGoldenMass<120 && zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 &&  abs(zGoldenDau1Eta)<2.1 &&  abs(zGoldenDau2Eta)<2.1 && zGoldenDau1Chi2<1000 && zGoldenDau2Chi2<1000 && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 &&(zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>=10 && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>=10 && zGoldenDau1NofMuonHits>0 && zGoldenDau2NofMuonHits>0 && ( zGoldenDau1HLTBit ==1 && zGoldenDau2HLTBit ==1)");

  TDirectory * dir = output_file->mkdir("goodZToMuMu2HLTPlots");
  dir->cd();
  TH1F * zMass = new TH1F("zMass", "zMass", 200, 0, 200);
  Events.Project("zMass", "zGoldenMass", kin_common + dau1TightWP1_notChi2AndTrackerMuon  + dau2TightWP1_notChi2AndTrackerMuon +  cut_1Iso + cut_2Iso + "zGoldenDau1HLTBit==1 && zGoldenDau2HLTBit==1" );

  zMass->Write();
  cout<<"Number of zGolden2HLT : "<<zMass->GetEntries()<<endl;
  delete zMass;
  output_file->cd("/");


 // zGoldenOneNotIso plots
 // TCut cut_zGoldenOneNotIso("zGoldenMass>60 && zGoldenMass<120 && zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 &&  abs(zGoldenDau1Eta)<2.1 &&  abs(zGoldenDau2Eta)<2.1 && zGoldenDau1Chi2<1000 && zGoldenDau2Chi2<1000 && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 &&(zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>=10 && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>=10 && zGoldenDau1NofMuonHits>0 && zGoldenDau2NofMuonHits>0 && ( zGoldenDau1HLTBit ==1 || zGoldenDau2HLTBit ==1)");
  TDirectory * dir = output_file->mkdir("oneNonIsolatedZToMuMuPlots");
  dir->cd();
  TH1F * zMass = new TH1F("zMass", "zMass", 200, 0, 200);
  Events.Project("zMass", "zGoldenMass", kin_common + dau1TightWP1_notChi2AndTrackerMuon  + dau2TightWP1_notChi2AndTrackerMuon + cut_OneNotIso);
  zMass->Write();
  cout<<"Number of zGoldenOneNotIso : "<<zMass->GetEntries()<<endl;
  delete zMass;
  output_file->cd("/");

//  // zGoldenTwoNotIso plots
//  TCut cut_zGoldenTwoNotIso("zGoldenMass>60 && zGoldenMass<120 && zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 && abs(zGoldenDau1Eta)<2.1 &&  abs(zGoldenDau2Eta)<2.1 && zGoldenDau1Chi2<1000 && zGoldenDau2Chi2<1000 && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 &&(zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>=10 && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>=10 && zGoldenDau1NofMuonHits>0 && zGoldenDau2NofMuonHits>0 && ( zGoldenDau1HLTBit ==1 || zGoldenDau2HLTBit ==1)");
  TDirectory * dir = output_file->mkdir("twoNonIsolatedZToMuMuPlots");
   dir->cd();
   TH1F * zMass = new TH1F("zMass", "zMass", 200, 0, 200);
   Events.Project("zMass","zGoldenMass", kin_common + dau1TightWP1_notChi2AndTrackerMuon  + dau2TightWP1_notChi2AndTrackerMuon + cut_TwoNotIso);
   zMass->Write();
  cout<<"Number of zGoldenTwoNotIso : "<<zMass->GetEntries()<<endl;
   delete zMass;
   output_file->cd("/");

 // zGoldenNotIso plots
   //  TCut cut_zGoldenNotIso("zGoldenMass>60 && zGoldenMass<120 && zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 &&  abs(zGoldenDau1Eta)<2.1 &&  abs(zGoldenDau2Eta)<2.1 && zGoldenDau1Chi2<1000 && zGoldenDau2Chi2<1000 && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 &&(zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>=10 && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>=10 && zGoldenDau1NofMuonHits>0 && zGoldenDau2NofMuonHits>0 && ( zGoldenDau1HLTBit ==1 || zGoldenDau2HLTBit ==1)");
  TDirectory * dir = output_file->mkdir("nonIsolatedZToMuMuPlots");
  dir->cd();
  TH1F * zMass = new TH1F("zMass", "zMass", 200, 0, 200);
  Events.Project("zMass", "zGoldenMass", kin_common + dau1TightWP1_notChi2AndTrackerMuon  + dau2TightWP1_notChi2AndTrackerMuon + cut_NotIso);
  zMass->Write() ;
  cout<<"Number of zGoldenNotIso : "<<zMass->GetEntries()<<endl;
  delete zMass;
  output_file->cd("/");

  // zGoldenSameCharge plots........
  TCut cut_zSameCharge("zSameChargeMass>60 && zSameChargeMass<120 && zSameChargeDau1Pt> 20 && zSameChargeDau2Pt>20 && zSameChargeDau1TrkIso< 3.0 && zSameChargeDau1TrkIso < 3.0 && abs(zSameChargeDau1Eta)<2.1 &&  abs(zSameChargeDau2Eta)<2.1 && zSameChargeDau1Chi2<1000 && zSameChargeDau2Chi2<1000 && abs(zSameChargeDau1dxyFromBS)<0.2 && abs(zSameChargeDau2dxyFromBS)<0.2 &&(zSameChargeDau1NofStripHits + zSameChargeDau1NofPixelHits)>10 && (zSameChargeDau2NofStripHits + zSameChargeDau2NofPixelHits)>10 && zSameChargeDau1NofMuonHits>0 && zSameChargeDau2NofMuonHits>0 && (zSameChargeDau1HLTBit==1 || zSameChargeDau2HLTBit==1) ");
  TDirectory * dir = output_file->mkdir("goodZToMuMuSameChargePlots");
  dir->cd();
  TH1F * zMass = new TH1F("zMass", "zMass", 200, 0, 200);
  Events.Project("zMass", "zSameChargeMass", cut_zSameCharge );
  zMass->Write();
  cout<<"Number of zGoldenSameCharge : "<<zMass->GetEntries()<<endl;
  delete zMass;
  output_file->cd("/");

 // zGoldenSameChargeNotIso plots........
  TCut cut_zSameChargeNotIso("zSameChargeMass>60 && zSameChargeMass<120 && zSameChargeDau1Pt> 20 && zSameChargeDau2Pt>20 && ( zSameChargeDau1TrkIso> 3.0 || zSameChargeDau1TrkIso > 3.0) && abs(zSameChargeDau1Eta)<2.1 &&  abs(zSameChargeDau2Eta)<2.1 && zSameChargeDau1Chi2<1000 && zSameChargeDau2Chi2<1000 && abs(zSameChargeDau1dxyFromBS)<0.2 && abs(zSameChargeDau2dxyFromBS)<0.2 &&(zSameChargeDau1NofStripHits + zSameChargeDau1NofPixelHits)>10 && (zSameChargeDau2NofStripHits + zSameChargeDau2NofPixelHits)>10 && zSameChargeDau1NofMuonHits>0 && zSameChargeDau2NofMuonHits>0 && (zSameChargeDau1HLTBit==1 || zSameChargeDau2HLTBit==1) ");
  TDirectory * dir = output_file->mkdir("goodZToMuMuSameChargeNotIsoPlots");
  dir->cd();
  TH1F * zMass = new TH1F("zMass", "zMass", 200, 0, 200);
  Events.Project("zMass", "zSameChargeMass", cut_zSameChargeNotIso );
  zMass->Write();
  cout<<"Number of zGoldenSameChargeNotIso : "<<zMass->GetEntries()<<endl;
  delete zMass;
  output_file->cd("/");

  
// zMuTrk plots
  TCut cut_zMuTrk("zGoldenMass@.size()==0  && zMuTrkDau1Pt> 20 && zMuTrkDau2Pt>20 &&  abs(zMuTrkDau1Eta)<2.1 &&  abs(zMuTrkDau2Eta)<2.1 && (zMuTrkDau1Chi2<1000) &&  ( zMuTrkDau2Chi2<1000) && abs(zMuTrkDau1dxyFromBS)<0.2 && abs(zMuTrkDau2dxyFromBS)<0.2 && ((zMuTrkDau1TrkNofStripHits + zMuTrkDau1TrkNofPixelHits)>10)  && ((zMuTrkDau2TrkNofStripHits + zMuTrkDau2TrkNofPixelHits)>10) && zMuTrkDau1TrkNofPixelHits>0 && zMuTrkDau2TrkNofPixelHits>0 &&  (zMuTrkDau1NofMuonHits>0 && zMuTrkDau1NofMuMatches>1) &&  (zMuTrkDau1HLTBit==1)");
  TDirectory * dir = output_file->mkdir("goodZToMuMuOneTrackPlots");
  dir->cd();
  TH1F * zMass = new TH1F("zMass", "zMass", 200, 0, 200);
  Events.Project("zMass", "zMuTrkMass", cut_zMuTrk +  cutTk_1Iso +  cutTk_2Iso );
 cout<<"Number of original zMuTrk : "<<zMass->GetEntries()<<endl; 

  // add events with when a golden does not pass the sta additional requirements 
TCut cut_zGoldenNotGoodSta("zGoldenDau1Pt>20 && zGoldenDau2Pt>20 && abs(zGoldenDau1Eta)<2.1 &&  abs(zGoldenDau2Eta)<2.1 && zGoldenDau1Chi2<10000 && zGoldenDau2Chi2<10000 && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10 && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10 && zGoldenDau1NofPixelHits>0 && zGoldenDau2NofPixelHits>0 && ( ((zGoldenDau1NofMuonHits==0 || zGoldenDau1NofMuMatches<2 ) && (zGoldenDau2HLTBit==1 && zGoldenDau2NofMuonHits>0 && zGoldenDau2NofMuMatches>1)) || ((zGoldenDau2NofMuonHits==0 || zGoldenDau2NofMuMatches<2 ) && (zGoldenDau1HLTBit==1 && zGoldenDau1NofMuonHits>0 && zGoldenDau1NofMuMatches>1)) )  && (zGoldenDau1HLTBit==1 || zGoldenDau2HLTBit==1) ");

  TH1F * zMass2 = new TH1F("zMass2", "zMass2", 200, 0, 200);
  Events.Project("zMass2", "zGoldenMass", cut_zGoldenNotGoodSta +  cut_1Iso +  cut_2Iso );

 cout<<"Number of zMuTrk from zGolden: "<<zMass2->GetEntries()<<endl; 
 zMass->Add(zMass2);
  zMass->Write();
  delete zMass;
  output_file->cd("/");

// zMuTrkMu plots

  TCut cut_zMuTrkMu("zMuTrkMuDau1Pt>20 && zMuTrkMuDau2Pt>20 && zMuTrkMuDau1TrkIso< 3.0 && zMuTrkMuDau2TrkIso<3.0 && abs(zMuTrkMuDau1Eta)<2.1 &&  abs(zMuTrkMuDau2Eta)<2.1 && (( zMuTrkMuDau1Chi2<1000 &&  zMuTrkMuDau1GlobalMuonBit==1 ) || ( zMuTrkMuDau2Chi2<1000 &&  zMuTrkMuDau2GlobalMuonBit==1))  && abs(zMuTrkMuDau1dxyFromBS)<0.2 && abs(zMuTrkMuDau2dxyFromBS)<0.2 && (( zMuTrkMuDau1GlobalMuonBit==1  && (zMuTrkMuDau1TrkNofStripHits + zMuTrkMuDau1TrkNofPixelHits)>10) || ( zMuTrkMuDau2GlobalMuonBit==1  && (zMuTrkMuDau2TrkNofStripHits + zMuTrkMuDau2TrkNofPixelHits)>10)  ) && (( zMuTrkMuDau1GlobalMuonBit==1 && zMuTrkMuDau1NofMuonHits>0) || (  zMuTrkMuDau2GlobalMuonBit==1 && zMuTrkMuDau2NofMuonHits>0 )) && ( (zMuTrkMuDau1HLTBit==1 && zMuTrkMuDau1GlobalMuonBit==1 ) || (zMuTrkMuDau2HLTBit==1 && zMuTrkMuDau2GlobalMuonBit==1 )) ");

  //   TCut cut_zMuTrkMu("zMuTrkMuMass>60 && zMuTrkMuMass<120 && zMuTrkMuDau1Pt> 20 && zMuTrkMuDau2Pt>20 && zMuTrkMuDau1TrkIso< 3.0 && zMuTrkMuDau2TrkIso < 3.0 && abs(zMuTrkMuDau1Eta)<2.1 &&  abs(zMuTrkMuDau2Eta)<2.1 && ((( zMuTrkMuDau1GlobalMuonBit==1 && zMuTrkMuDau1Chi2<1000 && zMuTrkMuDau1NofMuonHits>0) || ( zMuTrkMuDau1GlobalMuonBit==0 && zMuTrkMuDau1TrkChi2<1000)) || (( zMuTrkMuDau2GlobalMuonBit==1 && zMuTrkMuDau2Chi2<1000) || ( zMuTrkMuDau2GlobalMuonBit==0 && zMuTrkMuDau1TrkChi2<1000 && zMuTrkMuDau2NofMuonHits>0)) && abs(zMuTrkMuDau1dxyFromBS)<0.2 && abs(zMuTrkMuDau2dxyFromBS)<0.2 && ((zMuTrkMuDau1TrkNofStripHits + zMuTrkMuDau1TrkNofPixelHits)>=10)  && ((zMuTrkMuDau2TrkNofStripHits + zMuTrkMuDau2TrkNofPixelHits)>=10)) &&   (zMuTrkMuDau1HLTBit==1 || zMuTrkMuDau2HLTBit==1 ) ");
  TDirectory * dir = output_file->mkdir("goodZToMuMuOneTrackerMuonPlots");
  dir->cd();
  TH1F * zMass = new TH1F("zMass", "zMass", 200, 0, 200);
  Events.Project("zMass", "zMuTrkMuMass", cut_zMuTrkMu );

  Events.Project("zMass2", "zMuTrkMuMass", cut_zMuTrkMu );

  zMass->Write();
  cout<<"Number of zMuTrkMu : "<<zMass->GetEntries()<<endl;
  delete zMass;
  delete zMass2;

  output_file->cd("/");



  // zMuSta plots

 TCut cut_zMuSta("zGoldenMass@.size()==0 && zMuStaDau1Pt>20 && zMuStaDau2Pt>20 && abs(zMuStaDau1Eta)<2.1 &&  abs(zMuStaDau2Eta)<2.1 && abs(zMuStaDau1dxyFromBS)<0.2 && abs(zMuStaDau2dxyFromBS)<0.2 && ( (zMuStaDau1GlobalMuonBit==1  && (zMuStaDau1TrkNofStripHits + zMuStaDau1TrkNofPixelHits)>10 && zMuStaDau1TrkNofPixelHits>0) || ( zMuStaDau2GlobalMuonBit==1  && (zMuStaDau2TrkNofStripHits + zMuStaDau2TrkNofPixelHits)>10 && zMuStaDau2TrkNofPixelHits>0))  && (( zMuStaDau1SaNofMuonHits>0 && zMuStaDau1NofMuMatches>1 &&  zMuStaDau2SaNofMuonHits>0 && zMuStaDau2NofMuMatches>1 )) && ( (zMuStaDau1HLTBit==1 && zMuStaDau1GlobalMuonBit==1 ) || (zMuStaDau2HLTBit==1 && zMuStaDau2GlobalMuonBit==1 ))"); 

//  TCut cut_zMuSta("zMuStaMass>60 && zMuStaMass<120 && zMuStaDau1Pt> 20 && zMuStaDau2Pt>20 && zMuStaDau1TrkIso< 3.0 && zMuStaDau1TrkIso < 3.0 && abs(zMuStaDau1Eta)<2.1 &&  abs(zMuStaDau2Eta)<2.1  && abs(zMuStaDau1dxyFromBS)<0.2 && abs(zMuStaDau2dxyFromBS)<0.2 &&(zMuStaDau1NofStripHits + zMuStaDau1NofPixelHits)>=10 && (zMuStaDau2NofStripHits + zMuStaDau2NofPixelHits)>=10 && zMuStaDau1NofMuonHits>0 && zMuStaDau2NofMuonHits>0 && ((zMuStaDau1HLTBit==1  && zMuStaDau1GlobalMuonBit ==1 ) || ( zMuStaDau2HLTBit==1  && zMuStaDau2GlobalMuonBit ==1) ) ");
  TDirectory * dir = output_file->mkdir("goodZToMuMuOneStandAloneMuonPlots");
  dir->cd();
  TH1F * zMass = new TH1F("zMass", "zMass", 200, 0, 200);
  Events.Project("zMass", "zMuStaMass", cut_zMuSta  + cutSa_1Iso +  cutSa_2Iso );

  // add events from zGolden with not a good trk
TCut cut_zGoldenNotGoodTrk("zGoldenDau1Pt>20 && zGoldenDau2Pt>20 && abs(zGoldenDau1Eta)<2.1 &&  abs(zGoldenDau2Eta)<2.1 && zGoldenDau1Chi2<10000 && zGoldenDau2Chi2<10000 && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 && ( (((zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)<=10 ||  zGoldenDau1NofPixelHits==0) &&  (zGoldenDau2HLTBit==1 && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10 && zGoldenDau2NofPixelHits>0)) || (((zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)<=10 ||  zGoldenDau2NofPixelHits==0) &&  (zGoldenDau1HLTBit==1 && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10 && zGoldenDau1NofPixelHits>0))) &&  (zGoldenDau1NofMuonHits>0 && zGoldenDau2NofMuonHits>0 && zGoldenDau1NofMuMatches>1 && zGoldenDau2NofMuMatches>1) && (zGoldenDau1HLTBit==1 || zGoldenDau2HLTBit==1)");
  TH1F * zMass2 = new TH1F("zMass2", "zMass2", 200, 0, 200);
  Events.Project("zMass2", "zGoldenMassSa", cut_zGoldenNotGoodTrk  + cut_1Iso +  cut_2Iso );

  cout<<"Number of original zMuSta : "<<zMass->GetEntries()<<endl;
  cout<<"Number of additionl zMuSta from ZGolden : "<<zMass2->GetEntries()<<endl;
  zMass->Add(zMass2); 
 zMass->Write();

  delete zMass;
  delete zMass2;

  output_file->cd("/");




// zMuSta plots
  TCut cut_zMuMuSta("zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 && zGoldenDau1TrkIso< 3.0 && zGoldenDau1TrkIso < 3.0  && abs(zGoldenDau1Eta)<2.1 &&  abs(zGoldenDau2Eta)<2.1 && zGoldenDau1Chi2<1000 && zGoldenDau2Chi2<1000 && zGoldenDau1dxyFromBS<0.02 && zGoldenDau2dxyFromBS<0.02");
  TDirectory * dir = output_file->mkdir("zmumuSaMassHistogram");
  dir->cd();
  TH1F * zMass = new TH1F("zMass", "zMass", 200, 0, 200);
  Events.Project("zMass", "zGoldenMassSa", cut_zMuMuSta );
  zMass->Write();
  delete zMass;
  output_file->cd("/");

 
 
}
