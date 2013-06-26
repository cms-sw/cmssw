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




void plotsAfterCuts_OneLeg(){

  gStyle->SetOptStat();
  gROOT->SetStyle("Plain");
  using namespace std;

  //  #include <exception>;

  //    TFile *file = TFile::Open("rfio:/castor/cern.ch/user/d/degrutto/incl15WithBsPv/NtupleLoose_test_inclu15_1_2.root");
  // TFile *file = TFile::Open("../NutpleLooseTestNew_oneshot_all_10_1.root");
//TFile *file = TFile::Open("rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/NtupleLoose_test.root");
//    TTree * Events = dynamic_cast< TTree *> (file->Get("Events"));

  TChain Events("Events"); 
  // one need 130 events... each file has 1000 ev
 Events.Add("../zmmNtuple/NtupleLooseTestNew.root");
  /*  Events.Add("../zmmNtuple/NtupleLooseTestNew_oneshot_all_Test_1_None.root");
  Events.Add("../zmmNtuple/NtupleLooseTestNew_oneshot_all_Test_2_None.root");
  Events.Add("../zmmNtuple/NtupleLooseTestNew_oneshot_all_Test_3_None.root");
  Events.Add("../zmmNtuple/NtupleLooseTestNew_oneshot_all_Test_4_None.root");
  Events.Add("../zmmNtuple/NtupleLooseTestNew_oneshot_all_Test_5_None.root");
  Events.Add("../zmmNtuple/NtupleLooseTestNew_oneshot_all_Test_6_None.root");
  Events.Add("../zmmNtuple/NtupleLooseTestNew_oneshot_all_Test_7_None.root");
  Events.Add("../zmmNtuple/NtupleLooseTestNew_oneshot_all_Test_8_None.root");
  Events.Add("../zmmNtuple/NtupleLooseTestNew_oneshot_all_Test_9_None.root");
  Events.Add("../zmmNtuple/NtupleLooseTestNew_oneshot_all_Test_10_None.root");
  Events.Add("../zmmNtuple/NtupleLooseTestNew_oneshot_all_Test_11_None.root");
  Events.Add("../zmmNtuple/NtupleLooseTestNew_oneshot_all_Test_12_None.root");
  Events.Add("../zmmNtuple/NtupleLooseTestNew_oneshot_all_Test_13_None.root");
  */

  
  
  TFile * output_file = TFile::Open("histoZmm_OneCut.root", "RECREATE");
  // TFile * output_file = TFile::Open("histo_test.root", "RECREATE");
  
  // zGolden plots
  TCut cut_zGolden("zGoldenMass>60 && zGoldenMass<120 && zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 && zGoldenDau1TrkIso< 3.0 && zGoldenDau2TrkIso < 3.0 && abs(zGoldenDau1Eta)<2.1 &&  abs(zGoldenDau2Eta)<2.1 && (zGoldenDau1Chi2<10 || zGoldenDau2Chi2<10) && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 && ((zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>=10 || (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>=10) && (zGoldenDau1NofMuonHits>0 || zGoldenDau2NofMuonHits>0) && (zGoldenDau1HLTBit==1 || zGoldenDau2HLTBit==1) ");
  TDirectory * dir = output_file->mkdir("goodZToMuMuPlots");
  dir->cd();
  TH1F * zMass = new TH1F("zMass", "zMass", 200, 0, 200);
  //  Events.Draw("zGoldenMass");
  Events.Project("zMass", "zGoldenMass", cut_zGolden );
  cout<<"Number of zGoldenAA : "<<zMass->GetEntries()<<endl;
  setGraphics(zMass);
  zMass->Write();
  delete zMass;

  
  TCut cut_zGoldenPt15("zGoldenMass>20 && zGoldenDau1Pt> 15 && zGoldenDau2Pt>15 && zGoldenDau1TrkIso< 3.0 && zGoldenDau2TrkIso < 3.0  &&  abs(zGoldenDau1Eta)<2.1 &&  abs(zGoldenDau2Eta)<2.1 && (zGoldenDau1Chi2<10 || zGoldenDau2Chi2<10) && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 && ((zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>=10 || (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>=10) && (zGoldenDau1NofMuonHits>0 || zGoldenDau2NofMuonHits>0) && (zGoldenDau1HLTBit==1 || zGoldenDau2HLTBit==1)");
  dir->cd();
  
  TH1F * zMassPt15 = new TH1F("zMassPt15", "zMassPt15", 200, 0, 200);
  Events.Project("zMassPt15", "zGoldenMass", cut_zGoldenPt15  );
  setGraphics(zMassPt15);
  cout<<"Number of zGoldenPt15 : "<<zMassPt15->GetEntries()<<endl;
  zMassPt15->Write();
  delete zMassPt15;



  output_file->cd("/");
  
   TCut cut2_zGolden1HLT("zGoldenMass>60 && zGoldenMass<120 && zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 && zGoldenDau1TrkIso< 3.0 && zGoldenDau2TrkIso < 3.0 && abs(zGoldenDau1Eta)<2.1 &&  abs(zGoldenDau2Eta)<2.1 && (zGoldenDau1Chi2<10 || zGoldenDau2Chi2<10) && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 && ( (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>=10 || ((zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>=10)) && (zGoldenDau1NofMuonHits>0 || zGoldenDau2NofMuonHits>0) && ( zGoldenDau1HLTBit ==0 || zGoldenDau2HLTBit ==0)");
  TDirectory * dir = output_file->mkdir("goodZToMuMu1HLTPlots");
  dir->cd();
  TH1F * zMass = new TH1F("zMass", "zMass", 200, 0, 200);
    Events.Project("zMass", "zGoldenMass", cut2_zGolden1HLT );
  cout<<"Number of zGolden1HLT : "<<zMass->GetEntries()<<endl;
 zMass->Write();
  delete zMass;
 //zMass2->Write();

  output_file->cd("/");


  output_file->cd("/");
  
   TCut cut2_zGoldenAB1HLT("zGoldenMass>60 && zGoldenMass<120 && zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 && zGoldenDau1TrkIso< 3.0 && zGoldenDau2TrkIso < 3.0 && ( abs(zGoldenDau1Eta)>2.1 ||  abs(zGoldenDau2Eta)>2.1 )   && (zGoldenDau1Chi2<10 || zGoldenDau2Chi2<10) && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 && ((zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>=10 || (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>=10) && (zGoldenDau1NofMuonHits>0 || zGoldenDau2NofMuonHits>0) && ( zGoldenDau1HLTBit ==1 || zGoldenDau2HLTBit ==1)");
  TDirectory * dir = output_file->mkdir("goodZToMuMuAB1HLTPlots");
  dir->cd();
  TH1F * zMass = new TH1F("zMass", "zMass", 200, 0, 200);
    Events.Project("zMass", "zGoldenMass", cut2_zGoldenAB1HLT );
  cout<<"Number of zGoldenAB1HLT : "<<zMass->GetEntries()<<endl;
 zMass->Write();
  delete zMass;
 //zMass2->Write();

  output_file->cd("/");


   TCut cut2_zGolden2HLT("zGoldenMass>60 && zGoldenMass<120 && zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 && zGoldenDau1TrkIso< 3.0 && zGoldenDau2TrkIso < 3.0 && abs(zGoldenDau1Eta)<2.1 &&  abs(zGoldenDau2Eta)<2.1 && (zGoldenDau1Chi2<10 || zGoldenDau2Chi2<10) && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 && ((zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>=10 || (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>=10) && (zGoldenDau1NofMuonHits>0 || zGoldenDau2NofMuonHits>0) && ( zGoldenDau1HLTBit ==1 && zGoldenDau2HLTBit ==1)");

  TDirectory * dir = output_file->mkdir("goodZToMuMu2HLTPlots");
  dir->cd();
  TH1F * zMass = new TH1F("zMass", "zMass", 200, 0, 200);
  Events.Project("zMass", "zGoldenMass", cut2_zGolden2HLT );
  zMass->Write();
  cout<<"Number of zGolden2HLT : "<<zMass->GetEntries()<<endl;
  delete zMass;
  output_file->cd("/");


 // zGoldenOneNotIso plots
  TCut cut_zGoldenOneNotIso("zGoldenMass>60 && zGoldenMass<120 && zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 && ( (zGoldenDau1TrkIso> 3.0 &&  zGoldenDau2TrkIso < 3.0) || (zGoldenDau2TrkIso> 3.0 &&  zGoldenDau1TrkIso < 3.0))  && abs(zGoldenDau1Eta)<2.1 &&  abs(zGoldenDau2Eta)<2.1 && (zGoldenDau1Chi2<10 || zGoldenDau2Chi2<10) && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 && ((zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>=10 || (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>=10) && (zGoldenDau1NofMuonHits>0 || zGoldenDau2NofMuonHits>0) && ( zGoldenDau1HLTBit ==1 || zGoldenDau2HLTBit ==1)");
  TDirectory * dir = output_file->mkdir("oneNonIsolatedZToMuMuPlots");
  dir->cd();
  TH1F * zMass = new TH1F("zMass", "zMass", 200, 0, 200);
  Events.Project("zMass", "zGoldenMass", cut_zGoldenOneNotIso );
  zMass->Write();
  cout<<"Number of zGoldenOneNotIso : "<<zMass->GetEntries()<<endl;
  delete zMass;
  output_file->cd("/");

//  // zGoldenTwoNotIso plots
  TCut cut_zGoldenTwoNotIso("zGoldenMass>60 && zGoldenMass<120 && zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 && (zGoldenDau1TrkIso> 3.0 &&  zGoldenDau2TrkIso > 3.0) && abs(zGoldenDau1Eta)<2.1 &&  abs(zGoldenDau2Eta)<2.1 && (zGoldenDau1Chi2<10 || zGoldenDau2Chi2<10) && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 && ((zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>=10 || (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>=10) && (zGoldenDau1NofMuonHits>0 || zGoldenDau2NofMuonHits>0) && ( zGoldenDau1HLTBit ==1 || zGoldenDau2HLTBit ==1)");
  TDirectory * dir = output_file->mkdir("twoNonIsolatedZToMuMuPlots");
   dir->cd();
   TH1F * zMass = new TH1F("zMass", "zMass", 200, 0, 200);
   Events.Project("zMass","zGoldenMass", cut_zGoldenTwoNotIso );
   zMass->Write();
  cout<<"Number of zGoldenTwoNotIso : "<<zMass->GetEntries()<<endl;
   delete zMass;
   output_file->cd("/");

 // zGoldenNotIso plots
  TCut cut_zGoldenNotIso("zGoldenMass>60 && zGoldenMass<120 && zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 && (zGoldenDau1TrkIso> 3.0 ||  zGoldenDau2TrkIso > 3.0) && abs(zGoldenDau1Eta)<2.1 &&  abs(zGoldenDau2Eta)<2.1 && (zGoldenDau1Chi2<10 || zGoldenDau2Chi2<10) && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 && ((zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>=10 || (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>=10) && (zGoldenDau1NofMuonHits>0 || zGoldenDau2NofMuonHits>0) && ( zGoldenDau1HLTBit ==1 || zGoldenDau2HLTBit ==1)");
  TDirectory * dir = output_file->mkdir("nonIsolatedZToMuMuPlots");
  dir->cd();
  TH1F * zMass = new TH1F("zMass", "zMass", 200, 0, 200);
  Events.Project("zMass", "zGoldenMass", cut_zGoldenNotIso );
  zMass->Write() ;
  cout<<"Number of zGoldenNotIso : "<<zMass->GetEntries()<<endl;
  delete zMass;
  output_file->cd("/");

  // zGoldenSameCharge plots........
  TCut cut_zSameCharge("zSameChargeMass>60 && zSameChargeMass<120 && zSameChargeDau1Pt> 20 && zSameChargeDau2Pt>20 && zSameChargeDau1TrkIso< 3.0 && zSameChargeDau2TrkIso < 3.0 && abs(zSameChargeDau1Eta)<2.1 &&  abs(zSameChargeDau2Eta)<2.1 && zSameChargeDau1Chi2<10 && zSameChargeDau2Chi2<10 && abs(zSameChargeDau1dxyFromBS)<0.2 && abs(zSameChargeDau2dxyFromBS)<0.2 && ((zSameChargeDau1NofStripHits + zSameChargeDau1NofPixelHits)>=10 || (zSameChargeDau2NofStripHits + zSameChargeDau2NofPixelHits)>=10)  && (zSameChargeDau1NofMuonHits>0 || zSameChargeDau2NofMuonHits>0) && (zSameChargeDau1HLTBit==1 || zSameChargeDau2HLTBit==1) ");
  TDirectory * dir = output_file->mkdir("goodZToMuMuSameChargePlots");
  dir->cd();
  TH1F * zMass = new TH1F("zMass", "zMass", 200, 0, 200);
  Events.Project("zMass", "zSameChargeMass", cut_zSameCharge );
  zMass->Write();
  cout<<"Number of zGoldenSameCharge : "<<zMass->GetEntries()<<endl;
  delete zMass;
  output_file->cd("/");

 // zGoldenSameChargeNotIso plots........
  TCut cut_zSameChargeNotIso("zSameChargeMass>60 && zSameChargeMass<120 && zSameChargeDau1Pt> 20 && zSameChargeDau2Pt>20 && ( zSameChargeDau1TrkIso> 3.0 || zSameChargeDau2TrkIso > 3.0) && abs(zSameChargeDau1Eta)<2.1 &&  abs(zSameChargeDau2Eta)<2.1 && zSameChargeDau1Chi2<10 && zSameChargeDau2Chi2<10 && abs(zSameChargeDau1dxyFromBS)<0.2 && abs(zSameChargeDau2dxyFromBS)<0.2 && ( (zSameChargeDau1NofStripHits + zSameChargeDau1NofPixelHits)>=10 || (zSameChargeDau2NofStripHits + zSameChargeDau2NofPixelHits)>=10) && (zSameChargeDau1NofMuonHits>0 || zSameChargeDau2NofMuonHits>0) && (zSameChargeDau1HLTBit==1 || zSameChargeDau2HLTBit==1) ");
  TDirectory * dir = output_file->mkdir("goodZToMuMuSameChargeNotIsoPlots");
  dir->cd();
  TH1F * zMass = new TH1F("zMass", "zMass", 200, 0, 200);
  Events.Project("zMass", "zSameChargeMass", cut_zSameChargeNotIso );
  zMass->Write();
  cout<<"Number of zGoldenSameChargeNotIso : "<<zMass->GetEntries()<<endl;
  delete zMass;
  output_file->cd("/");

  
// zMuTrk plots
  TCut cut_zMuTrk("zMuTrkMass>60 && zMuTrkMass<120 && zMuTrkDau1Pt> 20 && zMuTrkDau2Pt>20 && zMuTrkDau1TrkIso< 3.0 && zMuTrkDau2TrkIso < 3.0 && abs(zMuTrkDau1Eta)<2.1 &&  abs(zMuTrkDau2Eta)<2.1 && (zMuTrkDau1Chi2<10) && abs(zMuTrkDau1dxyFromBS)<0.2 && abs(zMuTrkDau2dxyFromBS)<0.2 && ((zMuTrkDau1TrkNofStripHits + zMuTrkDau1TrkNofPixelHits)>=10) && (zMuTrkDau1NofMuonHits>0) &&  (zMuTrkDau1HLTBit==1) ");
  TDirectory * dir = output_file->mkdir("goodZToMuMuOneTrackPlots");
  dir->cd();
  TH1F * zMass = new TH1F("zMass", "zMass", 200, 0, 200);
  Events.Project("zMass", "zMuTrkMass", cut_zMuTrk );
  zMass->Write();
  cout<<"Number of zMuTrk : "<<zMass->GetEntries()<<endl;
  delete zMass;
  output_file->cd("/");

// zMuTrkMu plots
  TCut cut_zMuTrkMu("zMuTrkMuMass>60 && zMuTrkMuMass<120 && zMuTrkMuDau1Pt>20 && zMuTrkMuDau2Pt>20 && zMuTrkMuDau1TrkIso< 3.0 && zMuTrkMuDau2TrkIso<3.0 && abs(zMuTrkMuDau1Eta)<2.1 &&  abs(zMuTrkMuDau2Eta)<2.1 && (( zMuTrkMuDau1Chi2<10 &&  zMuTrkMuDau1GlobalMuonBit==1 ) || ( zMuTrkMuDau2Chi2<10 &&  zMuTrkMuDau2GlobalMuonBit==1))  && abs(zMuTrkMuDau1dxyFromBS)<0.2 && abs(zMuTrkMuDau2dxyFromBS)<0.2 && (( zMuTrkMuDau1GlobalMuonBit==1  && (zMuTrkMuDau1TrkNofStripHits + zMuTrkMuDau1TrkNofPixelHits)>=10) || ( zMuTrkMuDau2GlobalMuonBit==1  && (zMuTrkMuDau2TrkNofStripHits + zMuTrkMuDau2TrkNofPixelHits)>=10)  ) && (( zMuTrkMuDau1GlobalMuonBit==1 && zMuTrkMuDau1NofMuonHits>0) || (  zMuTrkMuDau2GlobalMuonBit==1 && zMuTrkMuDau2NofMuonHits>0 )) && ( (zMuTrkMuDau1HLTBit==1 && zMuTrkMuDau1GlobalMuonBit==1 ) || (zMuTrkMuDau2HLTBit==1 && zMuTrkMuDau2GlobalMuonBit==1 )) ");
  TDirectory * dir = output_file->mkdir("goodZToMuMuOneTrackerMuonPlots");
  dir->cd();
  TH1F * zMass = new TH1F("zMass", "zMass", 200, 0, 200);
  Events.Project("zMass", "zMuTrkMuMass", cut_zMuTrkMu );
  zMass->Write();
  cout<<"Number of zMuTrkMu : "<<zMass->GetEntries()<<endl;
  delete zMass;
  output_file->cd("/");



  // zMuSta plots
  TCut cut_zMuSta("zMuStaMass>60 && zMuStaMass<120 && zMuStaDau1Pt>20 && zMuStaDau2Pt>20 && zMuStaDau1TrkIso< 3.0 && zMuStaDau2TrkIso<3.0 && abs(zMuStaDau1Eta)<2.1 &&  abs(zMuStaDau2Eta)<2.1 && (( zMuStaDau1Chi2<10 &&  zMuStaDau1GlobalMuonBit==1 ) || ( zMuStaDau2Chi2<10 &&  zMuStaDau2GlobalMuonBit==1))  && abs(zMuStaDau1dxyFromBS)<0.2 && abs(zMuStaDau2dxyFromBS)<0.2 && (( zMuStaDau1GlobalMuonBit==1  && (zMuStaDau1TrkNofStripHits + zMuStaDau1TrkNofPixelHits)>=10) || ( zMuStaDau2GlobalMuonBit==1  && (zMuStaDau2TrkNofStripHits + zMuStaDau2TrkNofPixelHits)>=10)  ) && (( zMuStaDau1GlobalMuonBit==1 && zMuStaDau1NofMuonHits>0) || (  zMuStaDau2GlobalMuonBit==1 && zMuStaDau2NofMuonHits>0 )) && ( (zMuStaDau1HLTBit==1 && zMuStaDau1GlobalMuonBit==1 ) || (zMuStaDau2HLTBit==1 && zMuStaDau2GlobalMuonBit==1 )) ");

  TDirectory * dir = output_file->mkdir("goodZToMuMuOneStandAloneMuonPlots");
  dir->cd();
  TH1F * zMass = new TH1F("zMass", "zMass", 200, 0, 200);
  Events.Project("zMass", "zMuStaMass", cut_zMuSta );
  zMass->Write();
  cout<<"Number of zMuSta : "<<zMass->GetEntries()<<endl;
  delete zMass;
  output_file->cd("/");

// zMuSta plots
  TCut cut_zMuMuSta("zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 && zGoldenDau1TrkIso< 3.0 && zGoldenDau2TrkIso < 3.0  && zGoldenDau1Eta<2.1 &&  zGoldenDau2Eta<2.1 && (zGoldenDau1Chi2<10 || zGoldenDau2Chi2<10) && zGoldenDau1dxyFromBS<0.02 && zGoldenDau2dxyFromBS<0.02");
  TDirectory * dir = output_file->mkdir("zmumuSaMassHistogram");
  dir->cd();
  TH1F * zMass = new TH1F("zMass", "zMass", 200, 0, 200);
  Events.Project("zMass", "zGoldenMassSa", cut_zMuMuSta );
  zMass->Write();
  delete zMass;
  output_file->cd("/");

  
  TDirectory * dir = output_file->mkdir("zPlots");
  dir->cd();
  TH1F * zGoldenPt = new TH1F("zGoldenPt", "zGoldenPt", 200, 0, 200);
  Events.Project("zGoldenPt", "zGoldenPt", cut_zGolden );
  zGoldenPt->Write();
  delete zGoldenPt;
   

  TH1F * zGoldenY = new TH1F("zGoldenY", "zGoldenY", 200, -5, 5);
  Events.Project("zGoldenY", "zGoldenY", cut_zGolden );
  zGoldenY->Write();
  delete zGoldenY;
  
  output_file->cd("/");



  TDirectory * dir = output_file->mkdir("MuPlots");
  dir->cd();
  

  TH1F * muNotTriggeredEta = new TH1F("muNotTriggeredEta", "muNotTriggeredEta", 240, -6, 6.);
  TH1F * h2 = new TH1F("h2", "h2", 240, -6, 6.);
  
  Events.Project("muNotTriggeredEta","zGoldenDau1Eta", "zGoldenDau1HLTBit==0");
  Events.Project("h2","zGoldenDau2Eta", "zGoldenDau2HLTBit==0");

  muNotTriggeredEta->Add(h2);
  muNotTriggeredEta->Write();
  delete muNotTriggeredEta;
  delete h2;
  
  TH1F * zGoldenDauHighPt = new TH1F("zGoldenDauHighPt", "zGoldenDauHighPt", 200, 0, 200);
  Events.Project("zGoldenDauHighPt", "max(zGoldenDau1Pt, zGoldenDau2Pt)", cut_zGolden );
  zGoldenDauHighPt->Write();
  delete zGoldenDauHighPt;
  
  TH1F * zGoldenDauLowPt = new TH1F("zGoldenDauLowPt", "zGoldenDauLowPt", 200, 0, 200);
  Events.Project("zGoldenDauLowPt", "min(zGoldenDau1Pt, zGoldenDau2Pt)", cut_zGolden );
  zGoldenDauLowPt->Write();
  delete zGoldenDauLowPt;

  //(mu1.phi -mu2.phi)
  TH1F * deltaPhi = new TH1F("deltaPhi", "deltaPhi", 120, 0, 6.);
  TH1F * h2 = new TH1F("h2", "h2", 120, 0, 6. );
  TH1F * h3 = new TH1F("h3", "h3", 120, 0, 6. );
  
  /*    result = phi1 - phi2;
040     while (result > M_PI) result -= 2*M_PI;
041     while (result <= -M_PI) result += 2*M_PI;
042     return result;
  */ 

  Events.Project("deltaPhi", "abs(zGoldenDau1Phi - zGoldenDau2Phi)", "-TMath::Pi() < (zGoldenDau1Phi - zGoldenDau2Phi) < TMath::Pi()" , "zGoldenDau1Pt>20 && zGoldenDau2Pt >20"  + cut_zGolden);
  Events.Project("h2", "abs(zGoldenDau1Phi - zGoldenDau2Phi - 2 * TMath::Pi())", "(zGoldenDau1Phi - zGoldenDau2Phi) > TMath::Pi()" , "zGoldenDau1Pt>20 && zGoldenDau2Pt >20" + + cut_zGolden);
  Events.Project("h3", "abs(zGoldenDau1Phi - zGoldenDau2Phi + 2 * TMath::Pi())", "(zGoldenDau1Phi - zGoldenDau2Phi) <=  -TMath::Pi()", "zGoldenDau1Pt>20 && zGoldenDau2Pt >20" + cut_zGolden );
  
  deltaPhi->Add(h2, h3);
  deltaPhi->Write();
  delete deltaPhi;;
  delete h2;
  delete h3;
  
  

  // mu1.eta -mu2.eta
  TH1F * deltaEta = new TH1F("deltaEta", "deltaEta", 120, 0, 6.);
  Events.Project("deltaEta", "abs(zGoldenDau1Eta - zGoldenDau2Eta)", cut_zGolden );
  deltaEta->Write();
  delete deltaEta;
  
  TH1F * dua1Phi = new TH1F("dau1Phi", "dau1Phi", 120, -6, 6.);
  Events.Project("dau1Phi", "zGoldenDau1Phi" , cut_zGolden);
  dau1Phi->Write();
  delete dau1Phi;
  
  TH1F * dua2Phi = new TH1F("dau2Phi", "dau2Phi", 120, -6, 6.);
  Events.Project("dau2Phi", "zGoldenDau2Phi" , cut_zGolden);
  dau2Phi->Write();
  delete dau2Phi;
  
  TH1F * dau1Eta = new TH1F("dua1Eta", "dau1Eta", 120, -6, 6.);
  Events.Project("dau1Eta", "zGoldenDau1Eta", cut_zGolden );
  dau1Eta->Write();
  delete dau1Eta;
  
  TH1F * dau2Eta = new TH1F("dua2Eta", "dau2Eta", 120, -6, 6.);
  Events.Project("dau2Eta", "zGoldenDau2Eta" , cut_zGolden);
  dau2Eta->Write();
  delete dau2Eta;
  
  // quality variables
  // caveat: I'm  requiring isolations
  TH1F * dau1Chi2 = new TH1F("dua1Chi2", "dau1Chi2", 1000, 0, 100);
  Events.Project("dua1Chi2", "zGoldenDau1Chi2", cut_zGolden );
  dau1Chi2->Write();
  delete dau1Chi2;
  
  TH1F * dau2Chi2 = new TH1F("dua2Chi2", "dau2Chi2", 1000, 0, 100);
  Events.Project("dau2Chi2", "zGoldenDau2Chi2", cut_zGolden );
  dau2Chi2->Write();
  delete dau2Chi2;
  
  
  TH1F * dau1Dxy = new TH1F("dua1Dxy", "dau1Dxy", 500, 0, 5);
  Events.Project("dua1Dxy", "zGoldenDau1dxyFromBS", cut_zGolden );
  dau1Dxy->Write();
  delete dau1Dxy;
  
  TH1F * dau2Dxy = new TH1F("dua2Dxy", "dau2Dxy", 500, 0, 5);
  Events.Project("dua2Dxy", "zGoldenDau2dxyFromBS", cut_zGolden );
  dau2Dxy->Write();
  delete dau2Dxy;
  
   
  TH1F * dau1Dz= new TH1F("dua1Dz", "dau1Dz", 500, 0, 20);
  Events.Project("dua1Dz", "zGoldenDau1dzFromBS", cut_zGolden );
  dau1Dz->Write();
  delete dau1Dz;
  
  TH1F * dau2Dz = new TH1F("dua2Dz", "dau2Dz", 500, 0, 20);
  Events.Project("dua2Dz", "zGoldenDau2dzFromBS", cut_zGolden);
  dau2Dz->Write();
  delete dau2Dz;
  
  /*
  TH1F * dau1NofHit = new TH1F("dua1NofHit", "dau1NofHit", 100, -0.5, 99.5);
  Events.Project("dua1NofHit", "zGoldenDau1NofHit", cut_zGolden );
  dau1NofHit->Write();
  delete dau1NofHit;
  
  TH1F * dau2NofHit = new TH1F("dua2NofHit", "dau2NofHit", 100, -0.5, 99.5);
  Events.Project("dua2NofHit", "zGoldenDau2NofHit", cut_zGolden );
  dau2NofHit->Write();
  delete dau2NofHit;
  */ 
  
  TH1F * dau1NofMuCh = new TH1F("dua1NofMuCh", "dau1NofMuCh", 20, -0.5, 19.5);
  Events.Project("dua1NofMuCh", "zGoldenDau1NofMuChambers", cut_zGolden );
  dau1NofMuCh->Write();
  delete dau1NofMuCh;
  
  TH1F * dau2NofMuCh = new TH1F("dua2NofMuCh", "dau2NofMuCh", 20, -0.5, 19.5);
  Events.Project("dua2NofMuCh", "zGoldenDau2NofMuChambers", cut_zGolden );
  dau2NofMuCh->Write();
  delete dau2NofMuCh;
  
  
  TH1F * dau1NofMuMatches = new TH1F("dua1NofMuMatches", "dau1NofMuMatches", 20, -0.5, 19.5);
  Events.Project("dua1NofMuMatches", "zGoldenDau1NofMuMatches", cut_zGolden );
  dau1NofMuMatches->Write();
  delete dau1NofMuMatches;
  
  TH1F * dau2NofMuMatches = new TH1F("dua2NofMuMatches", "dau2NofMuMatches", 20, -0.5, 19.5);
  Events.Project("dua2NofMuMatches", "zGoldenDau2NofMuMatches", cut_zGolden );
  dau2NofMuMatches->Write();
  delete dau2NofMuMatches;
  
  TH1F * dau1EmEnergy  = new TH1F("dua1EmEnergy", "dau1EmEnergy", 200, -0.1, 19.9);
  Events.Project("dua1EmEnergy", "zGoldenDau1MuEnergyEm", cut_zGolden );
  dau1EmEnergy->Write();
  delete dau1EmEnergy;
  
  TH1F * dau2EmEnergy  = new TH1F("dua2EmEnergy", "dau2EmEnergy", 200, -0.1, 19.9);
  Events.Project("dua2EmEnergy", "zGoldenDau2MuEnergyEm", cut_zGolden );
  dau2EmEnergy->Write();
  delete dau2EmEnergy;
  
  TH1F * dau1HadEnergy  = new TH1F("dua1HadEnergy", "dau1HadEnergy", 200, -0.1, 19.9);
  Events.Project("dua1HadEnergy", "zGoldenDau1MuEnergyHad", cut_zGolden );
  dau1HadEnergy->Write();
  delete dau1HadEnergy;
  
  TH1F * dau2HadEnergy  = new TH1F("dua2HadEnergy", "dau2HadEnergy", 200, -0.1, 19.9);
  Events.Project("dua2HadEnergy", "zGoldenDau2MuEnergyHad", cut_zGolden );
  dau2HadEnergy->Write();
  delete dau2HadEnergy;
  

   

  TH2F * MuChambersVsMuMatches = new TH2F("MuChambersVsMuMatches", "MuChambersVsMuMatches", 21, -0.5, 20.5, 21, -0.5, 20.5);
  TH2F * hh2= new TH2F("hh2", "hh2", 21, -0.5, 20.5, 21, -0.5, 20.5);
  Events.Project("MuChambersVsMuMatches", "zGoldenDau1NofMuChambers:zGoldenDau1NofMuMatches", cut_zGolden);         
  Events.Project("hh2", "zGoldenDau2NofMuChambers:zGoldenDau2NofMuMatches", cut_zGolden);         
  
  MuChambersVsMuMatches->Add(hh2);
  MuChambersVsMuMatches->Write();
  MuChambersVsMuMatches->SetDrawOption("Box");
  //  MuChambersVsMuMatches->Draw("BOX");
  delete MuChambersVsMuMatches;
  delete hh2;
  output_file->cd("/");
  
  TDirectory * dir = output_file->mkdir("TrkPlots");
  dir->cd();
  
  
  
  /*  TH1F * nofHitTrk = new TH1F("nofHitTrk", "nofHitTrk", 100, -0.5, 99.5);
  Events.Project("nofHitTrk", "zMuTrkDau2NofHitTk", cut_zMuTrk );
  nofHitTrk->Write();
  delete nofHitTrk;
  */ 
  
  TH1F * trkChi2 = new TH1F("trkChi2", "trkChi2", 100, -0.5, 99.5);
  Events.Project("trkChi2", "zMuTrkDau2Chi2", cut_zMuTrk );
  trkChi2->Write();
  delete trkChi2;
  

  output_file->cd("/");
   
   
  
  TDirectory * dir = output_file->mkdir("StaPlots");
  dir->cd();
  
  // sta as zDaudxyFromBS=-1 by construction....

  TH1F *  staNofMuCh = new TH1F("staNofMuCh", "staNofMuCh", 20, -0.5, 19.5);
  TH1F * h2 = new TH1F("h2", "h2", 20, -0.5, 19.5);
  Events.Project("staNofMuCh", "zMuStaDau1NofMuChambers", cut_zMuSta + "zMuStaDau1dxyFromBS==-1" );
  Events.Project("h2", "zMuStaDau2NofMuChambers", cut_zMuSta + "zMuStaDau2dxyFromBS==-1" );
  staNofMuCh->Add(h2);
  staNofMuCh->Write();
  delete staNofMuCh; 
  delete h2; 
  
  TH1F *  staNofMuMatches = new TH1F("staNofMuMatches", "staNofMuMatches", 20, -0.5, 19.5);
  TH1F * h2 = new TH1F("h2", "h2", 20, -0.5, 19.5);
  Events.Project("staNofMuMatches", "zMuStaDau1NofMuMatches", cut_zMuSta + "zMuStaDau1dxyFromBS==-1" );
  Events.Project("h2", "zMuStaDau2NofMuMatches", cut_zMuSta + "zMuStaDau2dxyFromBS==-1" );
  staNofMuMatches->Add(h2);
  staNofMuMatches->Write();
  delete staNofMuMatches; 
  delete h2; 
  
  TH2F * MuChambersVsMuMatches= new TH2F("MuChambersVsMuMatches", "MuChambersVsMuMatches", 21, -0.5, 20.5, 21, -0.5, 20.5);
  TH2F * hh2= new TH2F("hh2", "hh2", 21, -0.5, 20.5, 21, -0.5, 20.5);
  Events.Project("MuChambersVsMuMatches", "zMuStaDau1NofMuChambers:zMuStaDau1NofMuMatches", cut_zMuSta);         
  Events.Project("hh2", "zMuStaDau2NofMuChambers:zMuStaDau2NofMuMatches", cut_zMuSta);         
  MuChambersVsMuMatches->Add(hh2);
  MuChambersVsMuMatches->SetDrawOption("Box");
  MuChambersVsMuMatches->Write();
  delete MuChambersVsMuMatches;
  delete hh2;
  output_file->cd("/");
  
  // isolations...
   TDirectory * dir = output_file->mkdir("IsoPlots");
   dir->cd();
   

   TH1F * TrkIsoPt20= new TH1F("TrkIsoPt20", "TrkIsoPt20", 200, 0, 20);
   TH1F * h2 = new TH1F("h2", "h2", 200, 0, 20);
   Events.Project("TkIsoPt20", "zGoldenDau1TrkIso" , "zGoldenDau2Pt>20");
   Events.Project("h2", "zGoldenDau2TrkIso", "zGoldenDau2Pt>20" );
   TrkIsoPt20->Add(h2);
   TrkIsoPt20->Write();
   delete TrkIsoPt20;
   delete h2;
   
   TH1F * EcalIsoPt20 = new TH1F("EcalIsoPt20", "EcalIsoPt20", 200, 0, 20);
   TH1F * h2 = new TH1F("h2", "h2", 200, 0, 20);
   Events.Project("TkIsoPt20", "zGoldenDau1EcalIso" , "zGoldenDau1Pt>20");
   Events.Project("h2", "zGoldenDau2EcalIso", "zGoldenDau2Pt>20" );
   EcalIsoPt20->Add(h2);
   EcalIsoPt20->Write();
   delete EcalIsoPt20;
   delete h2;
   
   TH1F * HcalIsoPt20 = new TH1F("HcalIsoPt20", "HcalIsoPt20", 200, 0, 20);
   TH1F * h2 = new TH1F("h2", "h2", 200, 0, 20);
   Events.Project("TkIsoPt20", "zGoldenDau1HcalIso" , "zGoldenDau1Pt>20" );
   Events.Project("h2", "zGoldenDau2HcalIso" , "zGoldenDau2Pt>20");
   HcalIsoPt20->Add(h2);
   HcalIsoPt20->Write();
   delete HcalIsoPt20;
   delete h2;
   
   TH1F * TrkIsoPt15= new TH1F("TrkIsoPt15", "TrkIsoPt15", 200, 0, 20);
   TH1F * h2 = new TH1F("h2", "h2", 200, 0, 20);
   Events.Project("TkIsoPt15", "zGoldenDau1TrkIso" , "zGoldenDau1Pt>20");
   Events.Project("h2", "zGoldenDau2TrkIso", "zGoldenDau2Pt>15" );
   TrkIsoPt15->Add(h2);
   TrkIsoPt15->Write();
   delete TrkIsoPt15;
   delete h2;

   TH1F * EcalIsoPt15 = new TH1F("EcalIsoPt15", "EcalIsoPt15", 200, 0, 20);
   TH1F * h2 = new TH1F("h2", "h2", 200, 0, 20);
   Events.Project("TkIsoPt15", "zGoldenDau1EcalIso" , "zGoldenDau1Pt>20");
   Events.Project("h2", "zGoldenDau2EcalIso", "zGoldenDau2Pt>15" );
   EcalIsoPt15->Add(h2);
   
   EcalIsoPt15->Write();
   delete EcalIsoPt15;
   delete h2;

   TH1F * HcalIsoPt15 = new TH1F("HcalIsoPt15", "HcalIsoPt15", 200, 0, 20);
   TH1F * h2 = new TH1F("h2", "h2", 200, 0, 20);
   Events.Project("TkIsoPt15", "zGoldenDau1HcalIso" , "zGoldenDau1Pt>20" );
   Events.Project("h2", "zGoldenDau2HcalIso" , "zGoldenDau2Pt>15");
   HcalIsoPt15->Add(h2);

   HcalIsoPt15->Write();
   delete HcalIsoPt15;
   delete h2;
   output_file->cd("/");



   output_file->Close();
 
 
}
