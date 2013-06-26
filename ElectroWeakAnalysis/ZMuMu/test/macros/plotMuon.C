#include "TFile.h"
#include "TChain.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TTree.h"
#include "TAxis.h"
#include "TCanvas.h"
//#if !defined(__CINT__) && !defined(__MAKECINT__)                                                                                                          
#include <string>
#include <iostream>
#include <sstream>
//#endif   









void setGraphicsDATA(TH1F *histo){
gStyle->SetOptStat(0);
  histo->SetMarkerColor(kBlack);
  histo->SetMarkerStyle(20);
  histo->SetMarkerSize(0.8);
  histo->SetLineWidth(2);
  histo->SetLineColor(kBlack);
 }





void setGraphicsMC(TH1F *histo){
gStyle->SetOptStat(0);
  histo->SetFillColor(kAzure+7);
  histo->SetLineWidth(2);
  histo->SetLineWidth(2); 
  histo->SetLineColor(kBlue+1); 

}
   



void compareDATAMC( string name, TH1F * data, TH1F * mc , bool setLogY=kFALSE){
  TCanvas c;
  gStyle->SetOptStat(0);
  mc->Sumw2();
  double scale =1;
  if (mc->Integral()>0) scale = data->Integral()/  mc->Integral();
  mc->Scale(scale);
  mc->SetMaximum(data->GetMaximum()*1.5 + 1);
  mc->Draw("HIST");
  data->Draw("esame");
  data->SetTitle(name.c_str());
  mc->SetTitle(name.c_str());
  leg = new TLegend(0.65,0.60,0.85,0.75);
  leg->SetFillColor(kWhite);
  leg->AddEntry(data,"data");
  leg->AddEntry(mc,"MC","f");
  leg->Draw();
  string plotname= name + ".gif";
  c.SetLogy(setLogY);
  c.SetTitle(name.c_str());
  c.SaveAs(plotname.c_str());
  leg->Clear(); 


}




void plotMuon(){

gStyle->SetOptStat(0);
gROOT->SetStyle("Plain");
using namespace std;

TChain * chainDATA = new TChain("Events"); // create the chain with tree "T"

//chainMC.Add("MinBias2010_MC/NtupleLooseTestNew_oneshot_all_MCMinBias.root");
//  chainMC.Add("MinBias2010_MC/NtupleLoose_all_15apr_23_0.root");
//  chainMC.Add("MinBias2010_MC/NtupleLoose_all_15apr_23_1.root");


 int  nFiles = 151;

  
for(int j=1;j<nFiles;++j){
    ostringstream oss;
    oss<<j;
    //string name= "zmmNtuple/NtupleLooseTestNew_oneshot_all_Test_"+oss.str()+"_None.root";
    string name= "../runAll_muAnal_v7/res/MuNtuples_"+oss.str()+"_1.root";
    cout << name << endl;  
    chainDATA->Add(name.c_str());
  }



TChain * chainMC = new TChain("Events"); 

 nFiles =220;

  
for(int j=1;j<nFiles;++j){
    ostringstream oss;
    oss<<j;
    //string name= "zmmNtuple/NtupleLooseTestNew_oneshot_all_Test_"+oss.str()+"_None.root";
    string name= "../runAll_muAnal_ppmuX_v4/res/MuNtuples_"+oss.str()+"_1.root";
    cout << name << endl;  
    chainMC->Add(name.c_str());
  }

 


TFile * out = new TFile("histoMuons.root", "RECREATE");


 TCut qualCut ("MuGlobalMuonBit==1 && MuTrackerMuonBit==1 && (MuNofStripHits + MuNofPixelHits)>=10 && MuNofMuonHits>0 &&  MuChi2<10 && MuTrkIso<3.0 && abs(MuEta)<2.1 && MuPt>10 &&  abs(MuDxyFromBS)<0.2" );


 TCut qualCutButIso ("MuGlobalMuonBit==1 && MuTrackerMuonBit==1 && (MuNofStripHits + MuNofPixelHits)>=10 && MuNofMuonHits>0 &&  MuChi2<10 && MuTrkIso<1000.0 && abs(MuEta)<2.1 && MuPt>10 &&  abs(MuDxyFromBS)<0.2" );

 TCut noQualCut ("MuGlobalMuonBit==1 && MuNofStripHits>=1 && MuNofMuonHits>-1 &&  MuChi2<1000 && MuTrkIso<1000.0 && abs(MuEta)<2.1 && MuPt>10 &&  abs(MuDxyFromBS)<0.2" );



TCut stdCut ("MuGlobalMuonBit==1 && MuTrkIso<3.0 && abs(MuEta)<2.1 &&  MuPt>10 &&  abs(MuDxyFromBS)<0.2" );

 TCut qualCutHLT ("MuGlobalMuonBit==1 && MuTrackerMuonBit==1 && (MuNofStripHits + MuNofPixelHits)>=10 && MuNofMuonHits>0 &&  MuChi2<10 && MuTrkIso<3.0 && MuHLTBit==1 && abs(MuEta)<2.1 &&abs(MuDxyFromBS)<0.2" );

 TCut qualCutButIsoHLT ("MuGlobalMuonBit==1 && MuTrackerMuonBit==1 && (MuNofStripHits + MuNofPixelHits)>=10 && MuNofMuonHits>0 &&  MuChi2<10 && MuTrkIso<1000.0 && MuHLTBit==1 && abs(MuEta)<2.1 &&abs(MuDxyFromBS)<0.2" );

 TCut noQualCutButHLT ("MuGlobalMuonBit==1 && (MuNofStripHits + MuNofPixelHits)>=1 && MuNofMuonHits>-2 &&  MuChi2<1000 && MuTrkIso<1000.0 && MuHLTBit==1 && abs(MuEta)<2.1 &&abs(MuDxyFromBS)<0.2" );



 TCut qualCutNoHLT ("MuGlobalMuonBit==1 && MuTrackerMuonBit==1 && (MuNofStripHits + MuNofPixelHits)>=10 && MuNofMuonHits>0 &&  MuChi2<10 && MuTrkIso<3.0 && abs(MuEta)<2.1 && MuHLTBit==0 &&abs(MuDxyFromBS)<0.2 && MuPt>10" );

 //==============   histo Data ===============///

TDirectory * dir = out->mkdir("MuDATAPlots");
  dir->cd();

   
 TH1F * muPtDATADen= new TH1F("muPtDATADen", "muPtDATADen",  100, 0, 100);
 chainDATA->Project("muPtDATADen", "MuPt", qualCut);
 setGraphicsDATA(muPtDATADen);
   muPtDATADen->Write();
   delete muPtDATADen;

 TH1F * muEtaDATADen= new TH1F("muEtaDATADen", "muEtaDATADen",  100, -2.5, 2.5);
 chainDATA->Project("muEtaDATADen", "MuEta", qualCut);
 setGraphicsDATA(muEtaDATADen);
   muEtaDATADen->Write();
   delete muEtaDATADen;




 TH1F * muPtDATADenNotIso= new TH1F("muPtDATADenNotIso", "muPtDATADenNotIso",  100, 0, 100);
 chainDATA->Project("muPtDATADenNotIso", "MuPt", qualCutButIso);
 setGraphicsDATA(muPtDATADenNotIso);
   muPtDATADenNotIso->Write();
   delete muPtDATADenNotIso;



 TH1F * muPtDATADenNoCut= new TH1F("muPtDATADenNoCut", "muPtDATADenNoCut",  100, 0, 100);
 chainDATA->Project("muPtDATADenNoCut", "MuPt", noQualCut);
 setGraphicsDATA(muPtDATADenNoCut);
   muPtDATADenNoCut->Write();
   delete muPtDATADenNoCut;







  
 TH1F * muEtaDATA= new TH1F("muEtaDATA", "muEtaDATA",  100, -2.5, 2.5);
 chainDATA->Project("muEtaDATA", "MuEta", qualCut);
 setGraphicsDATA(muEtaDATA);
   muEtaDATA->Write();
   delete muEtaDATA;


 TH1F * muPtDATANum= new TH1F("muPtDATANum", "muPtDATANum",  100, 0, 100);
 chainDATA->Project("muPtDATANum", "MuPt", qualCutHLT);
 setGraphicsDATA(muPtDATANum);
   muPtDATANum->Write();
   delete muPtDATANum;

 TH1F * muEtaDATANum= new TH1F("muEtaDATANum", "muEtaDATANum",  100, -2.5, 2.5);
 chainDATA->Project("muEtaDATANum", "MuEta", qualCutHLT + "MuPt>10");
 setGraphicsDATA(muEtaDATANum);
   muEtaDATANum->Write();
   delete muEtaDATANum;




 TH1F * muPtDATANumNotIso= new TH1F("muPtDATANumNotIso", "muPtDATANumNotIso",  100, 0, 100);
 chainDATA->Project("muPtDATANumNotIso", "MuPt", qualCutButIsoHLT);
 setGraphicsDATA(muPtDATANumNotIso);
   muPtDATANumNotIso->Write();
   delete muPtDATANumNotIso;



 TH1F * muPtDATANumNoCut= new TH1F("muPtDATANumNoCut", "muPtDATANumNoCut",  100, 0, 100);
 chainDATA->Project("muPtDATANumNoCut", "MuPt", noQualCutButHLT);
 setGraphicsDATA(muPtDATANumNoCut);
   muPtDATANumNoCut->Write();
   delete muPtDATANumNoCut;




   TH2F * muPtEtaDATADen= new TH2F("muPtEtaDATADen", "muPtEtaDATADenEta",   100, -2.5, 2.5, 100, 0, 100);
 chainDATA->Project("muPtEtaDATADen", "MuPt:MuEta", qualCut);
 //setGraphics2dimDATA(muPtEtaDATADen);
   muPtEtaDATADen->Write();
   delete muPtEtaDATADen;


   TH2F * muPtEtaDATANum= new TH2F("muPtEtaDATANum", "muPtEtaDATANum",  100, -2.5, 2.5,  100, 0, 100);
 chainDATA->Project("muPtEtaDATANum", "MuPt:MuEta", qualCutHLT);
 // setGraphics2dimDATA(muPtEtaDATANum);
   muPtEtaDATANum->Write();
   delete muPtEtaDATANum;



   TH1F * muNoHLTEta= new TH1F("muNoHLTEta", "muNoHLTEta", 100, -2.5, 2.5);
 chainDATA->Project("muNoHLTEta", "MuEta", qualCutNoHLT);
 setGraphicsDATA(muNoHLTEta);
   muNoHLTEta->Write();
   delete muNoHLTEta;

   TH1F * muNoHLTChi2= new TH1F("muNoHLTChi2", "muNoHLTChi2", 100, 0, 100);
 chainDATA->Project("muNoHLTChi2", "MuChi2", qualCutNoHLT);
 setGraphicsDATA(muNoHLTChi2);
   muNoHLTChi2->Write();
   delete muNoHLTChi2;



   TH1F * muNoHLTPixelHits= new TH1F("muNoHLTPixelHits", "muNoHLTPixelHits", 10, -0.5, 9.5);
 chainDATA->Project("muNoHLTPixelHits", "MuNofPixelHits", qualCutNoHLT);
 setGraphicsDATA(muNoHLTPixelHits);
 muNoHLTPixelHits->Write();
   delete muNoHLTPixelHits;


   TH1F * muNoHLTMuonHits= new TH1F("muNoHLTMuonHits", "muNoHLTMuonHits", 30, -0.5, 39.5);
 chainDATA->Project("muNoHLTMuonHits", "MuNofMuonHits", qualCutNoHLT);
 setGraphicsDATA(muNoHLTMuonHits);
 muNoHLTMuonHits->Write();
   delete muNoHLTMuonHits;

   TH1F * muNoHLTMuMatches= new TH1F("muNoHLTMuMatches", "muNoHLTMuMatches", 15, -0.5, 14.5);
 chainDATA->Project("muNoHLTMuMatches", "MuNofMuMatches", qualCutNoHLT);
 setGraphicsDATA(muNoHLTMuMatches);
 muNoHLTMuMatches->Write();
   delete muNoHLTMuMatches;




   TH1F * muNoHLTMuEnergyEm= new TH1F("muNoHLTMuEnergyEm", "muNoHLTMuEnergyEm", 100, 0, 10);
 chainDATA->Project("muNoHLTMuEnergyEm", "MuEnergyEm", qualCutNoHLT);
 setGraphicsDATA(muNoHLTMuEnergyEm);
 muNoHLTMuEnergyEm->Write();
   delete muNoHLTMuEnergyEm;


   TH1F * muNoHLTMuEnergyHad= new TH1F("muNoHLTMuEnergyHad", "muNoHLTMuEnergyHad", 100, 0, 20);
 chainDATA->Project("muNoHLTMuEnergyHad", "MuEnergyHad", qualCutNoHLT);
 setGraphicsDATA(muNoHLTMuEnergyHad);
 muNoHLTMuEnergyHad->Write();
   delete muNoHLTMuEnergyHad;






  TH1F * muNofPixelHits= new TH1F("muNofPixelHits", "muNofPixelHits", 10, -0.5, 9.5);
 chainDATA->Project("muNofPixelHits", "MuNofPixelHits", stdCut);
 setGraphicsDATA(muNofPixelHits);
   muNofPixelHits->Write();
   delete muNofPixelHits;

  TH1F * muNofStripHits= new TH1F("muNofStripHits", "muNofStripHits", 30, -0.5, 29.5);
 chainDATA->Project("muNofStripHits", "MuNofStripHits", stdCut);
 setGraphicsDATA(muNofStripHits);
   muNofStripHits->Write();
   delete muNofStripHits;


  TH1F * muNofMuonHits= new TH1F("muNofMuonHits", "muNofMuonHits", 40, -0.5, 39.5);
 chainDATA->Project("muNofMuonHits", "MuNofMuonHits", stdCut);
 setGraphicsDATA(muNofMuonHits);
   muNofMuonHits->Write();
   delete muNofMuonHits;


  TH1F * muNofMuMatches= new TH1F("muNofMuMatches", "muNofMuMatches", 15, -0.5, 14.5);
 chainDATA->Project("muNofMuMatches", "MuNofMuMatches", stdCut);
 setGraphicsDATA(muNofMuMatches);
   muNofMuMatches->Write();
   delete muNofMuMatches;




  TH1F * muEnergyEm= new TH1F("muEnergyEm", "muEnergyEm", 100, 0, 10);
 chainDATA->Project("muEnergyEm", "MuEnergyEm", stdCut);
 setGraphicsDATA(muEnergyEm);
   muEnergyEm->Write();
   delete muEnergyEm;


  TH1F * muEnergyHad= new TH1F("muEnergyHad", "muEnergyHad", 100, 0, 20);
 chainDATA->Project("muEnergyHad", "MuEnergyHad", stdCut);
 setGraphicsDATA(muEnergyHad);
   muEnergyHad->Write();
   delete muEnergyHad;




  TH1F * muChi2= new TH1F("muChi2", "muChi2", 100, 0, 100);
 chainDATA->Project("muChi2", "MuChi2", stdCut);
 setGraphicsDATA(muChi2);
   muChi2->Write();
   delete muChi2;


   /// after quality cuts


  TH1F * muNofPixelHitsAfterCut= new TH1F("muNofPixelHitsAfterCut", "muNofPixelHits", 10, -0.5, 9.5);
 chainDATA->Project("muNofPixelHitsAfterCut", "MuNofPixelHits", "MuGlobalMuonBit==1 && MuTrackerMuonBit==1 && MuNofStripHits>=7 && MuNofMuonHits>0 &&  MuChi2<10 && MuTrkIso<3.0 && MuHLTBit==1 && abs(MuEta)<2.1 &&abs(MuDxyFromBS)<0.2");
 setGraphicsDATA(muNofPixelHitsAfterCut);
   muNofPixelHitsAfterCut->Write();
   delete muNofPixelHitsAfterCut;

  TH1F * muNofStripHitsAfterCut= new TH1F("muNofStripHitsAfterCut", "muNofStripHitsAfterCut", 30, -0.5, 29.5);
 chainDATA->Project("muNofStripHitsAfterCut", "MuNofStripHits", "MuGlobalMuonBit==1 && MuTrackerMuonBit==1 && MuNofPixelHits>0 && MuNofMuonHits>0 &&  MuChi2<10 && MuTrkIso<3.0 && MuHLTBit==1 && abs(MuEta)<2.1 && MuPt>10 && abs(MuDxyFromBS)<0.2");
 setGraphicsDATA(muNofStripHitsAfterCut);
   muNofStripHitsAfterCut->Write();
   delete muNofStripHitsAfterCut;


  TH1F * muNofMuonHitsAfterCut= new TH1F("muNofMuonHitsAfterCut", "muNofMuonHitsAfterCut", 15, -0.5, 14.5);
 chainDATA->Project("muNofMuonHitsAfterCut", "MuNofMuonHits", "MuGlobalMuonBit==1 && MuTrackerMuonBit==1 && (MuNofStripHits + MuNofPixelHits)>=10 && MuNofMuonHits>-1 &&  MuChi2<10 && MuTrkIso<3.0 && MuHLTBit==1 && abs(MuEta)<2.1 && MuPt>10 && abs(MuDxyFromBS)<0.2");
 setGraphicsDATA(muNofMuonHitsAfterCut);
   muNofMuonHitsAfterCut->Write();
   delete muNofMuonHitsAfterCut;


  TH1F * muNofMuMatchesAfterCut= new TH1F("muNofMuMatchesAfterCut", "muNofMuMatchesAfterCut", 15, -0.5, 14.5);
 chainDATA->Project("muNofMuMatchesAfterCut", "MuNofMuMatches", "MuGlobalMuonBit==1 && MuTrackerMuonBit==1 && (MuNofStripHits + MuNofPixelHits)>=10 && MuNofMuonHits>-1 &&  MuChi2<10 && MuTrkIso<3.0 && MuHLTBit==1 && abs(MuEta)<2.1 && MuPt>10 && abs(MuDxyFromBS)<0.2");
 setGraphicsDATA(muNofMuMatchesAfterCut);
   muNofMuMatchesAfterCut->Write();
   delete muNofMuMatchesAfterCut;




  TH1F * muEnergyEmAfterCut= new TH1F("muEnergyEmAfterCut", "muEnergyEmAfterCut", 100, 0, 10);
 chainDATA->Project("muEnergyEmAfterCut", "MuEnergyEm", qualCutHLT);
 setGraphicsDATA(muEnergyEmAfterCut);
   muEnergyEmAfterCut->Write();
   delete muEnergyEmAfterCut;


  TH1F * muEnergyHadAfterCut= new TH1F("muEnergyHadAfterCut", "muEnergyHadAfterCut", 100, 0, 20);
 chainDATA->Project("muEnergyHadAfterCut", "MuEnergyHad", qualCutHLT);
 setGraphicsDATA(muEnergyHadAfterCut);
   muEnergyHadAfterCut->Write();
   delete muEnergyHadAfterCut;




  TH1F * muChi2AfterCut= new TH1F("muChi2AfterCut", "muChi2AfterCut", 100, 0, 100);
 chainDATA->Project("muChi2AfterCut", "MuChi2", "MuGlobalMuonBit==1 && MuTrackerMuonBit==1 && (MuNofStripHits + MuNofPixelHits)>=10 && MuNofMuonHits>-1 &&  MuChi2<1000 && MuTrkIso<3.0 && MuHLTBit==1 && abs(MuEta)<2.1 && MuPt>10 && abs(MuDxyFromBS)<0.2");
 setGraphicsDATA(muChi2AfterCut);
   muChi2AfterCut->Write();
   delete muChi2AfterCut;










out->cd("/");



//==============   histo MC ===============///
TDirectory * dir = out->mkdir("MuMCPlots");
  dir->cd();

  TH1F * muPtMCDen= new TH1F("muPtMCDen", "muPtMCDen",  100, 0, 100);
 chainMC->Project("muPtMCDen", "MuPt", qualCut);
 setGraphicsMC(muPtMCDen);
   muPtMCDen->Write();
   delete muPtMCDen;


 TH1F * muEtaMCDen= new TH1F("muEtaMCDen", "muEtaMCDen",  100, -2.5, 2.5);
 chainMC->Project("muEtaMCDen", "MuEta", qualCut);
 setGraphicsMC(muEtaMCDen);
   muEtaMCDen->Write();
   delete muEtaMCDen;



 TH1F * muPtMCDenNotIso= new TH1F("muPtMCDenNotIso", "muPtMCDenNotIso",  100, 0, 100);
 chainMC->Project("muPtMCDenNotIso", "MuPt", qualCutButIso);
 setGraphicsMC(muPtMCDenNotIso);
   muPtMCDenNotIso->Write();
   delete muPtMCDenNotIso;



 TH1F * muPtMCDenNoCut= new TH1F("muPtMCDenNoCut", "muPtMCDenNoCut",  100, 0, 100);
 chainMC->Project("muPtMCDenNoCut", "MuPt", noQualCut);
 setGraphicsMC(muPtMCDenNoCut);
   muPtMCDenNoCut->Write();
   delete muPtMCDenNoCut;





TH1F * muEtaMC= new TH1F("muEtaMC", "muEtaMC",  100, -2.5, 2.5);
 chainMC->Project("muEtaMC", "MuEta", qualCut);
 setGraphicsMC(muEtaMC);
   muEtaMC->Write();
   delete muEtaMC;

 TH1F * muPtMCNum= new TH1F("muPtMCNum", "muPtMCNum",  100, 0, 100);
 chainMC->Project("muPtMCNum", "MuPt", qualCutHLT);
 setGraphicsMC(muPtMCNum);
   muPtMCNum->Write();
   delete muPtMCNum;

 TH1F * muEtaMCNum= new TH1F("muEtaMCNum", "muEtaMCNum",  100, -2.5, 2.5);
 chainMC->Project("muEtaMCNum", "MuEta", qualCutHLT + "MuPt>10");
 setGraphicsMC(muEtaMCNum);
   muEtaMCNum->Write();
   delete muEtaMCNum;



 TH1F * muPtMCNumNotIso= new TH1F("muPtMCNumNotIso", "muPtMCNumNotIso",  100, 0, 100);
 chainMC->Project("muPtMCNumNotIso", "MuPt", qualCutButIsoHLT);
 setGraphicsMC(muPtMCNumNotIso);
   muPtMCNumNotIso->Write();
   delete muPtMCNumNotIso;



 TH1F * muPtMCNumNoCut= new TH1F("muPtMCNumNoCut", "muPtMCNumNoCut",  100, 0, 100);
 chainMC->Project("muPtMCNumNoCut", "MuPt", noQualCutButHLT);
 setGraphicsMC(muPtMCNumNoCut);
   muPtMCNumNoCut->Write();
   delete muPtMCNumNoCut;



   TH2F * muPtEtaMCDen= new TH2F("muPtEtaMCDen", "muPtEtaMCDenEta",   100, -2.5, 2.5, 100, 0, 100);
 chainMC->Project("muPtEtaMCDen", "MuPt:MuEta", qualCut);
 //setGraphics2dimMC(muPtEtaMCDen);
   muPtEtaMCDen->Write();
   delete muPtEtaMCDen;


   TH2F * muPtEtaMCNum= new TH2F("muPtEtaMCNum", "muPtEtaMCNum",  100, -2.5, 2.5,  100, 0, 100);
 chainMC->Project("muPtEtaMCNum", "MuPt:MuEta", qualCutHLT);
 // setGraphics2dimMC(muPtEtaMCNum);
   muPtEtaMCNum->Write();
   delete muPtEtaMCNum;



   TH1F * muNoHLTEta= new TH1F("muNoHLTEta", "muNoHLTEta", 100, -2.5, 2.5);
 chainMC->Project("muNoHLTEta", "MuEta", qualCutNoHLT);
 setGraphicsMC(muNoHLTEta);
   muNoHLTEta->Write();
   delete muNoHLTEta;


   TH1F * muNoHLTChi2= new TH1F("muNoHLTChi2", "muNoHLTChi2", 100, 0, 100);
 chainMC->Project("muNoHLTChi2", "MuChi2", qualCutNoHLT);
 setGraphicsMC(muNoHLTChi2);
   muNoHLTChi2->Write();
   delete muNoHLTChi2;



   TH1F * muNoHLTPixelHits= new TH1F("muNoHLTPixelHits", "muNoHLTPixelHits", 10, -0.5, 9.5);
 chainMC->Project("muNoHLTPixelHits", "MuNofPixelHits", qualCutNoHLT);
 setGraphicsMC(muNoHLTPixelHits);
 muNoHLTPixelHits->Write();
   delete muNoHLTPixelHits;


   TH1F * muNoHLTMuonHits= new TH1F("muNoHLTMuonHits", "muNoHLTMuonHits", 40, -0.5, 39.5);
 chainMC->Project("muNoHLTMuonHits", "MuNofMuonHits", qualCutNoHLT);
 setGraphicsMC(muNoHLTMuonHits);
 muNoHLTMuonHits->Write();
   delete muNoHLTMuonHits;


   TH1F * muNoHLTMuMatches= new TH1F("muNoHLTMuMatches", "muNoHLTMuMatches", 15, -0.5, 14.5);
 chainMC->Project("muNoHLTMuMatches", "MuNofMuMatches", qualCutNoHLT);
 setGraphicsMC(muNoHLTMuMatches);
 muNoHLTMuMatches->Write();
   delete muNoHLTMuMatches;



   TH1F * muNoHLTMuEnergyEm= new TH1F("muNoHLTMuEnergyEm", "muNoHLTMuEnergyEm", 100, 0, 10);
 chainMC->Project("muNoHLTMuEnergyEm", "MuEnergyEm", qualCutNoHLT);
 setGraphicsMC(muNoHLTMuEnergyEm);
 muNoHLTMuEnergyEm->Write();
   delete muNoHLTMuEnergyEm;


   TH1F * muNoHLTMuEnergyHad= new TH1F("muNoHLTMuEnergyHad", "muNoHLTMuEnergyHad", 100, 0, 20);
 chainMC->Project("muNoHLTMuEnergyHad", "MuEnergyHad", qualCutNoHLT);
 setGraphicsMC(muNoHLTMuEnergyHad);
 muNoHLTMuEnergyHad->Write();
   delete muNoHLTMuEnergyHad;




  TH1F * muNofPixelHits= new TH1F("muNofPixelHits", "muNofPixelHits", 10, -0.5, 9.5);
 chainMC->Project("muNofPixelHits", "MuNofPixelHits", stdCut);
 setGraphicsMC(muNofPixelHits);
   muNofPixelHits->Write();
   delete muNofPixelHits;

  TH1F * muNofStripHits= new TH1F("muNofStripHits", "muNofStripHits", 30, -0.5, 29.5);
 chainMC->Project("muNofStripHits", "MuNofStripHits", stdCut);
 setGraphicsMC(muNofStripHits);
   muNofStripHits->Write();
   delete muNofStripHits;


  TH1F * muNofMuonHits= new TH1F("muNofMuonHits", "muNofMuonHits", 40, -0.5, 39.5);
 chainMC->Project("muNofMuonHits", "MuNofMuonHits", stdCut);
 setGraphicsMC(muNofMuonHits);
   muNofMuonHits->Write();
   delete muNofMuonHits;


  TH1F * muNofMuMatches= new TH1F("muNofMuMatches", "muNofMuMatches", 15, -0.5, 14.5);
 chainMC->Project("muNofMuMatches", "MuNofMuMatches", stdCut);
 setGraphicsMC(muNofMuMatches);
   muNofMuMatches->Write();
   delete muNofMuMatches;



  TH1F * muEnergyEm= new TH1F("muEnergyEm", "muEnergyEm", 100, 0, 10);
 chainMC->Project("muEnergyEm", "MuEnergyEm", stdCut);
 setGraphicsMC(muEnergyEm);
   muEnergyEm->Write();
   delete muEnergyEm;


  TH1F * muEnergyHad= new TH1F("muEnergyHad", "muEnergyHad", 100, 0, 20);
 chainMC->Project("muEnergyHad", "MuEnergyHad", stdCut);
 setGraphicsMC(muEnergyHad);
   muEnergyHad->Write();
   delete muEnergyHad;




  TH1F * muChi2= new TH1F("muChi2", "muChi2", 100, 0, 100);
 chainMC->Project("muChi2", "MuChi2", stdCut);
 setGraphicsMC(muChi2);
   muChi2->Write();
   delete muChi2;


   /// after quality cuts


  TH1F * muNofPixelHitsAfterCut= new TH1F("muNofPixelHitsAfterCut", "muNofPixelHits", 10, -0.5, 9.5);
 chainMC->Project("muNofPixelHitsAfterCut", "MuNofPixelHits", "MuGlobalMuonBit==1 && MuTrackerMuonBit==1 && MuNofStripHits>=7 && MuNofMuonHits>0 &&  MuChi2<10 && MuTrkIso<3.0 && MuHLTBit==1 && abs(MuEta)<2.1 &&abs(MuDxyFromBS)<0.2");
 setGraphicsMC(muNofPixelHitsAfterCut);
   muNofPixelHitsAfterCut->Write();
   delete muNofPixelHitsAfterCut;

  TH1F * muNofStripHitsAfterCut= new TH1F("muNofStripHitsAfterCut", "muNofStripHitsAfterCut", 30, -0.5, 29.5);
 chainMC->Project("muNofStripHitsAfterCut", "MuNofStripHits", "MuGlobalMuonBit==1 && MuTrackerMuonBit==1 && MuNofPixelHits>0 && MuNofMuonHits>0 &&  MuChi2<10 && MuTrkIso<3.0 && MuHLTBit==1 && abs(MuEta)<2.1 && MuPt>10  &&abs(MuDxyFromBS)<0.2");
 setGraphicsMC(muNofStripHitsAfterCut);
   muNofStripHitsAfterCut->Write();
   delete muNofStripHitsAfterCut;


  TH1F * muNofMuonHitsAfterCut= new TH1F("muNofMuonHitsAfterCut", "muNofMuonHitsAfterCut", 40, -0.5, 39.5);
 chainMC->Project("muNofMuonHitsAfterCut", "MuNofMuonHits", "MuGlobalMuonBit==1 && MuTrackerMuonBit==1 && (MuNofStripHits + MuNofPixelHits)>=10 && MuNofMuonHits>-1 &&  MuChi2<10 && MuTrkIso<3.0 && MuHLTBit==1 && abs(MuEta)<2.1 && MuPt>10 && abs(MuDxyFromBS)<0.2");
 setGraphicsMC(muNofMuonHitsAfterCut);
   muNofMuonHitsAfterCut->Write();
   delete muNofMuonHitsAfterCut;




  TH1F * muNofMuMatchesAfterCut= new TH1F("muNofMuMatchesAfterCut", "muNofMuMatchesAfterCut", 15, -0.5, 14.5);
 chainMC->Project("muNofMuMatchesAfterCut", "MuNofMuMatches", "MuGlobalMuonBit==1 && MuTrackerMuonBit==1 && (MuNofStripHits + MuNofPixelHits)>=10 && MuNofMuonHits>-1 &&  MuChi2<10 && MuTrkIso<3.0 && MuHLTBit==1 && abs(MuEta)<2.1 && MuPt>10 && abs(MuDxyFromBS)<0.2");
 setGraphicsMC(muNofMuMatchesAfterCut);
   muNofMuMatchesAfterCut->Write();
   delete muNofMuMatchesAfterCut;




  TH1F * muEnergyEmAfterCut= new TH1F("muEnergyEmAfterCut", "muEnergyEmAfterCut", 100, 0, 10);
 chainMC->Project("muEnergyEmAfterCut", "MuEnergyEm", qualCutHLT);
 setGraphicsMC(muEnergyEmAfterCut);
   muEnergyEmAfterCut->Write();
   delete muEnergyEmAfterCut;


  TH1F * muEnergyHadAfterCut= new TH1F("muEnergyHadAfterCut", "muEnergyHadAfterCut", 100, 0, 20);
 chainMC->Project("muEnergyHadAfterCut", "MuEnergyHad", qualCutHLT);
 setGraphicsMC(muEnergyHadAfterCut);
   muEnergyHadAfterCut->Write();
   delete muEnergyHadAfterCut;




  TH1F * muChi2AfterCut= new TH1F("muChi2AfterCut", "muChi2AfterCut", 100, 0, 100);
 chainMC->Project("muChi2AfterCut", "MuChi2", "MuGlobalMuonBit==1 && MuTrackerMuonBit==1 && (MuNofStripHits + MuNofPixelHits)>=10 && MuNofMuonHits>-1 &&  MuChi2<1000 && MuTrkIso<3.0 && MuHLTBit==1 && abs(MuEta)<2.1 && MuPt>10 && abs(MuDxyFromBS)<0.2");
 setGraphicsMC(muChi2AfterCut);
   muChi2AfterCut->Write();
   delete muChi2AfterCut;









out->cd("/");




//==============   histo DATA vs MC ===============///
TDirectory * dir = out->mkdir("DATAvsMCPlots");
  dir->cd();

  TH1F * muPtDATADen= (TH1F*) out->Get("MuDATAPlots/muPtDATADen");
  TH1F * muPtMCDen= (TH1F*) out->Get("MuMCPlots/muPtMCDen");  
  compareDATAMC( "PtDen", muPtDATADen, muPtMCDen, kTRUE );
 
  TH1F * muPtDATANum= (TH1F*) out->Get("MuDATAPlots/muPtDATANum");
  TH1F * muPtMCNum= (TH1F*) out->Get("MuMCPlots/muPtMCNum");  
  compareDATAMC( "PtNum", muPtDATANum, muPtMCNum , kTRUE);

  TH1F * muEtaDATADen= (TH1F*) out->Get("MuDATAPlots/muEtaDATADen");
  TH1F * muEtaMCDen= (TH1F*) out->Get("MuMCPlots/muEtaMCDen");  
  compareDATAMC( "EtaDen", muEtaDATADen, muEtaMCDen , kTRUE);
 
  TH1F * muEtaDATANum= (TH1F*) out->Get("MuDATAPlots/muEtaDATANum");
  TH1F * muEtaMCNum= (TH1F*) out->Get("MuMCPlots/muEtaMCNum");  
  compareDATAMC( "EtaNum", muEtaDATANum, muEtaMCNum , kTRUE);




  TH1F * muPtDATADenNotIso= (TH1F*) out->Get("MuDATAPlots/muPtDATADenNotIso");
  TH1F * muPtMCDenNotIso= (TH1F*) out->Get("MuMCPlots/muPtMCDenNotIso");  
  compareDATAMC( "PtDenNotIso", muPtDATADenNotIso, muPtMCDenNotIso, kTRUE );
 
  TH1F * muPtDATANumNotIso= (TH1F*) out->Get("MuDATAPlots/muPtDATANumNotIso");
  TH1F * muPtMCNumNotIso= (TH1F*) out->Get("MuMCPlots/muPtMCNumNotIso");  
  compareDATAMC( "PtNumNotIso", muPtDATANumNotIso, muPtMCNumNotIso , kTRUE);



  TH1F * muPtDATADenNoCut= (TH1F*) out->Get("MuDATAPlots/muPtDATADenNoCut");
  TH1F * muPtMCDenNoCut= (TH1F*) out->Get("MuMCPlots/muPtMCDenNoCut");  
  compareDATAMC( "PtDenNoCut", muPtDATADenNoCut, muPtMCDenNoCut , kTRUE);
 
  TH1F * muPtDATANumNoCut= (TH1F*) out->Get("MuDATAPlots/muPtDATANumNoCut");
  TH1F * muPtMCNumNoCut= (TH1F*) out->Get("MuMCPlots/muPtMCNumNoCut");  
  compareDATAMC( "PtNumNoCut", muPtDATANumNoCut, muPtMCNumNoCut , kTRUE);


 
  
  TH1F * muEtaDATA= (TH1F*) out->Get("MuDATAPlots/muEtaDATA");
  TH1F * muEtaMC= (TH1F*) out->Get("MuMCPlots/muEtaMC");  
  compareDATAMC( "Eta", muEtaDATA, muEtaMC , kTRUE);

   TH1F * muNoHLTEtaDATA= (TH1F*) out->Get("MuDATAPlots/muNoHLTEta");
  TH1F * muNoHLTEtaMC= (TH1F*) out->Get("MuMCPlots/muNoHLTEta");  
  compareDATAMC( "NoHLTEta", muNoHLTEtaDATA, muNoHLTEtaMC, kTRUE );


   TH1F * muNoHLTChi2DATA= (TH1F*) out->Get("MuDATAPlots/muNoHLTChi2");
  TH1F * muNoHLTChi2MC= (TH1F*) out->Get("MuMCPlots/muNoHLTChi2");  
  compareDATAMC( "NoHLTChi2", muNoHLTChi2DATA, muNoHLTChi2MC, kTRUE );




   TH1F * muNoHLTPixelHitsDATA= (TH1F*) out->Get("MuDATAPlots/muNoHLTPixelHits");
  TH1F * muNoHLTPixelHitsMC= (TH1F*) out->Get("MuMCPlots/muNoHLTPixelHits");  
  compareDATAMC( "NoHLTPixelHits", muNoHLTPixelHitsDATA, muNoHLTPixelHitsMC );



   TH1F * muNoHLTMuonHitsDATA= (TH1F*) out->Get("MuDATAPlots/muNoHLTMuonHits");
  TH1F * muNoHLTMuonHitsMC= (TH1F*) out->Get("MuMCPlots/muNoHLTMuonHits");  
  compareDATAMC( "NoHLTMuonHits", muNoHLTMuonHitsDATA, muNoHLTMuonHitsMC );

   TH1F * muNoHLTMuMatchesDATA= (TH1F*) out->Get("MuDATAPlots/muNoHLTMuMatches");
  TH1F * muNoHLTMuMatchesMC= (TH1F*) out->Get("MuMCPlots/muNoHLTMuMatches");  
  compareDATAMC( "NoHLTMuMatches", muNoHLTMuMatchesDATA, muNoHLTMuMatchesMC );




   TH1F * muNoHLTMuEnergyEmDATA= (TH1F*) out->Get("MuDATAPlots/muNoHLTMuEnergyEm");
  TH1F * muNoHLTMuEnergyEmMC= (TH1F*) out->Get("MuMCPlots/muNoHLTMuEnergyEm");  
  compareDATAMC( "NoHLTMuEnergyEm", muNoHLTMuEnergyEmDATA, muNoHLTMuEnergyEmMC , kTRUE);


   TH1F * muNoHLTMuEnergyHadDATA= (TH1F*) out->Get("MuDATAPlots/muNoHLTMuEnergyHad");
  TH1F * muNoHLTMuEnergyHadMC= (TH1F*) out->Get("MuMCPlots/muNoHLTMuEnergyHad");  
  compareDATAMC( "NoHLTMuEnergyHad", muNoHLTMuEnergyHadDATA, muNoHLTMuEnergyHadMC , kTRUE);



   TH1F * muPixelHitsDATA= (TH1F*) out->Get("MuDATAPlots/muNofPixelHits");
  TH1F * muPixelHitsMC= (TH1F*) out->Get("MuMCPlots/muNofPixelHits");  
  compareDATAMC( "PixelHits", muPixelHitsDATA, muPixelHitsMC );



   TH1F * muStripHitsDATA= (TH1F*) out->Get("MuDATAPlots/muNofStripHits");
  TH1F * muStripHitsMC= (TH1F*) out->Get("MuMCPlots/muNofStripHits");  
  compareDATAMC( "StripHits", muStripHitsDATA, muStripHitsMC );


   TH1F * muMuonHitsDATA= (TH1F*) out->Get("MuDATAPlots/muNofMuonHits");
  TH1F * muMuonHitsMC= (TH1F*) out->Get("MuMCPlots/muNofMuonHits");  
  compareDATAMC( "MuonHits", muMuonHitsDATA, muMuonHitsMC );


   TH1F * muMuMatchesDATA= (TH1F*) out->Get("MuDATAPlots/muNofMuMatches");
  TH1F * muMuMatchesMC= (TH1F*) out->Get("MuMCPlots/muNofMuMatches");  
  compareDATAMC( "MuMatches", muMuMatchesDATA, muMuMatchesMC );



   TH1F * muEnergyEmDATA= (TH1F*) out->Get("MuDATAPlots/muEnergyEm");
  TH1F * muEnergyEmMC= (TH1F*) out->Get("MuMCPlots/muEnergyEm");  
  compareDATAMC( "MuEnergyEm", muEnergyEmDATA, muEnergyEmMC , kTRUE);


TH1F * muEnergyHadDATA = (TH1F*) out->Get("MuDATAPlots/muEnergyHad");
  TH1F * muEnergyHadMC= (TH1F*) out->Get("MuMCPlots/muEnergyHad");  
  compareDATAMC( "MuEnergyHad", muEnergyHadDATA, muEnergyHadMC , kTRUE);



   TH1F * muChi2DATA= (TH1F*) out->Get("MuDATAPlots/muChi2");
  TH1F * muChi2MC= (TH1F*) out->Get("MuMCPlots/muChi2");  
  compareDATAMC( "Chi2", muChi2DATA, muChi2MC , kTRUE);


  // after quality cuts



   TH1F * muPixelHitsAfterCutDATA= (TH1F*) out->Get("MuDATAPlots/muNofPixelHitsAfterCut");
  TH1F * muPixelHitsAfterCutMC= (TH1F*) out->Get("MuMCPlots/muNofPixelHitsAfterCut");  
  compareDATAMC( "PixelHitsAfterCut", muPixelHitsAfterCutDATA, muPixelHitsAfterCutMC );



   TH1F * muStripHitsAfterCutDATA= (TH1F*) out->Get("MuDATAPlots/muNofStripHitsAfterCut");
  TH1F * muStripHitsAfterCutMC= (TH1F*) out->Get("MuMCPlots/muNofStripHitsAfterCut");  
  compareDATAMC( "StripHitsAfterCut", muStripHitsAfterCutDATA, muStripHitsAfterCutMC );


   TH1F * muMuonHitsAfterCutDATA= (TH1F*) out->Get("MuDATAPlots/muNofMuonHitsAfterCut");
  TH1F * muMuonHitsAfterCutMC= (TH1F*) out->Get("MuMCPlots/muNofMuonHitsAfterCut");  
  compareDATAMC( "MuonHitsAfterCut", muMuonHitsAfterCutDATA, muMuonHitsAfterCutMC );


   TH1F * muMuMatchesAfterCutDATA= (TH1F*) out->Get("MuDATAPlots/muNofMuMatchesAfterCut");
  TH1F * muMuMatchesAfterCutMC= (TH1F*) out->Get("MuMCPlots/muNofMuMatchesAfterCut");  
  compareDATAMC( "MuMatchesAfterCut", muMuMatchesAfterCutDATA, muMuMatchesAfterCutMC );



   TH1F * muEnergyEmAfterCutDATA= (TH1F*) out->Get("MuDATAPlots/muEnergyEmAfterCut");
  TH1F * muEnergyEmAfterCutMC= (TH1F*) out->Get("MuMCPlots/muEnergyEmAfterCut");  
  compareDATAMC( "MuEnergyEmAfterCut", muEnergyEmAfterCutDATA, muEnergyEmAfterCutMC , kTRUE);

  
TH1F * muEnergyHadAfterCutDATA = (TH1F*) out->Get("MuDATAPlots/muEnergyHadAfterCut");
  TH1F * muEnergyHadAfterCutMC= (TH1F*) out->Get("MuMCPlots/muEnergyHadAfterCut");  
  compareDATAMC( "MuEnergyHadAfterCut", muEnergyHadAfterCutDATA, muEnergyHadAfterCutMC , kTRUE);
  


   TH1F * muChi2AfterCutDATA= (TH1F*) out->Get("MuDATAPlots/muChi2AfterCut");
  TH1F * muChi2AfterCutMC= (TH1F*) out->Get("MuMCPlots/muChi2AfterCut");  
  compareDATAMC( "Chi2AfterCut", muChi2AfterCutDATA, muChi2AfterCutMC , kTRUE);


  TH2F * muPtEtaDATANum= (TH2F*) out->Get("MuDATAPlots/muPtEtaDATANum");
  TH2F * muPtEtaDATADen= (TH2F*) out->Get("MuDATAPlots/muPtEtaDATADen");


  muPtEtaDATANum->Divide(muPtEtaDATADen);                              
       gStyle->SetPalette(1,0);
      
       muPtEtaDATANum->SetTitle("HLT_Mu9 pt vs eta efficiency");
       muPtEtaDATANum->GetYaxis()->SetRangeUser(9,49);
       TCanvas c1;       
       muPtEtaDATANum->Draw("COLZ");      

       c1.SaveAs("MuHLT_Mu9_PtvsEtaEfficiency.gif");



   

out->cd("/");








  

 out->Close();


}
