#include "TFile.h"
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
  histo->SetMarkerColor(kBlack);
  histo->SetMarkerStyle(20);
  histo->SetMarkerSize(0.8);
  histo->SetLineWidth(2);
  histo->SetLineColor(kBlack);
 }





void setGraphicsMC(TH1F *histo){
  histo->SetFillColor(kAzure+7);
  histo->SetLineWidth(2);
  histo->SetLineWidth(2); 
  histo->SetLineColor(kBlue+1); 

}
   



void compareDATAMC( string name, TH1F * data, TH1F * mc ){
  TCanvas c;
   mc->Sumw2();
  double scale = data->Integral()/  mc->Integral();
  mc->Scale(scale);
  mc->SetMaximum(data->GetMaximum()*1.5 + 1);
  mc->Draw("HIST");
  data->Draw("esame");
  leg = new TLegend(0.65,0.60,0.85,0.75);
  leg->SetFillColor(kWhite);
  leg->AddEntry(data,"data");
  leg->AddEntry(mc,"MC","f");
  leg->Draw();
  string plotname= name + ".gif"; 
  c.SaveAs(plotname.c_str());
  leg->Clear(); 


}




void plotMuon(){

gStyle->SetOptStat();
gROOT->SetStyle("Plain");
using namespace std;

TChain chainDATA("Events"); // create the chain with tree "T"

//chainMC.Add("MinBias2010_MC/NtupleLooseTestNew_oneshot_all_MCMinBias.root");
//  chainMC.Add("MinBias2010_MC/NtupleLoose_all_15apr_23_0.root");
//  chainMC.Add("MinBias2010_MC/NtupleLoose_all_15apr_23_1.root");

chainDATA.Add("MuonsNtuple.root");

TChain chainMC("Events"); 
chainMC.Add("MuonsNtuple_1_MinBias.root"); 
 
TFile out("histoMuons.root", "RECREATE");


 TCut qualCut ("MuGlobalMuonBit==1 && MuTrackerMuonBit==1 && MuNofStripHits>=10 && MuNofMuonHits>0 &&  MuChi2<10 && MuTrkIso<3.0 && abs(MuEta)<2.1 &&abs(MuDxyFromBS)<0.2" );

TCut stdCut ("MuGlobalMuonBit==1 && MuTrkIso<3.0 && abs(MuEta)<2.1 &&abs(MuDxyFromBS)<0.2" );

 TCut qualCutHLT ("MuGlobalMuonBit==1 && MuTrackerMuonBit==1 && MuNofStripHits>=10 && MuNofMuonHits>0 &&  MuChi2<10 && MuTrkIso<3.0 && MuHLTBit==1 && abs(MuEta)<2.1 &&abs(MuDxyFromBS)<0.2" );

 TCut qualCutNoHLT ("MuGlobalMuonBit==1 && MuTrackerMuonBit==1 && MuNofStripHits>=10 && MuNofMuonHits>0 &&  MuChi2<10 && MuTrkIso<3.0 && abs(MuEta)<2.1 && MuHLTBit==0 &&abs(MuDxyFromBS)<0.2" );


 //==============   histo Data ===============///
 TDirectory * dir = out.mkdir("MuDATAPlots");
  dir->cd();

 TH1F * muPtDATADen= new TH1F("muPtDATADen", "muPtDATADen",  100, 0, 100);
 chainDATA.Project("muPtDATADen", "MuPt", qualCut);
 setGraphicsDATA(muPtDATADen);
   muPtDATADen->Write();
   delete muPtDATADen;

 TH1F * muEtaDATA= new TH1F("muEtaDATA", "muEtaDATA",  100, -2.5, 2.5);
 chainDATA.Project("muEtaDATA", "MuEta", qualCut);
 setGraphicsDATA(muEtaDATA);
   muEtaDATA->Write();
   delete muEtaDATA;


 TH1F * muPtDATANum= new TH1F("muPtDATANum", "muPtDATANum",  100, 0, 100);
 chainDATA.Project("muPtDATANum", "MuPt", qualCutHLT);
 setGraphicsDATA(muPtDATANum);
   muPtDATANum->Write();
   delete muPtDATANum;


   TH2F * muPtEtaDATADen= new TH2F("muPtEtaDATADen", "muPtEtaDATADenEta",   100, -2.5, 2.5, 100, 0, 100);
 chainDATA.Project("muPtEtaDATADen", "MuPt:MuEta", qualCut);
 //setGraphics2dimDATA(muPtEtaDATADen);
   muPtEtaDATADen->Write();
   delete muPtEtaDATADen;


   TH2F * muPtEtaDATANum= new TH2F("muPtEtaDATANum", "muPtEtaDATANum",  100, -2.5, 2.5,  100, 0, 100);
 chainDATA.Project("muPtEtaDATANum", "MuPt:MuEta", qualCutHLT);
 // setGraphics2dimDATA(muPtEtaDATANum);
   muPtEtaDATANum->Write();
   delete muPtEtaDATANum;



   TH1F * muNoHLTEta= new TH1F("muNoHLTEta", "muNoHLTEta", 100, -2.5, 2.5);
 chainDATA.Project("muNoHLTEta", "MuEta", qualCutNoHLT);
 setGraphicsDATA(muNoHLTEta);
   muNoHLTEta->Write();
   delete muNoHLTEta;

   TH1F * muNoHLTPixelHits= new TH1F("muNoHLTPixelHits", "muNoHLTPixelHits", 10, -0.5, 9.5);
 chainDATA.Project("muNoHLTPixelHits", "MuNofPixelHits", qualCutNoHLT);
 setGraphicsDATA(muNoHLTPixelHits);
 muNoHLTPixelHits->Write();
   delete muNoHLTPixelHits;

  TH1F * muNofPixelHits= new TH1F("muNofPixelHits", "muNofPixelHits", 10, -0.5, 9.5);
 chainDATA.Project("muNofPixelHits", "MuNofPixelHits", stdCut);
 setGraphicsDATA(muNofPixelHits);
   muNofPixelHits->Write();
   delete muNofPixelHits;

  TH1F * muNofStripHits= new TH1F("muNofStripHits", "muNofStripHits", 30, -0.5, 29.5);
 chainDATA.Project("muNofStripHits", "MuNofStripHits", stdCut);
 setGraphicsDATA(muNofStripHits);
   muNofStripHits->Write();
   delete muNofStripHits;


  TH1F * muNofMuonHits= new TH1F("muNofMuonHits", "muNofMuonHits", 40, -0.5, 39.5);
 chainDATA.Project("muNofMuonHits", "MuNofMuonHits", stdCut);
 setGraphicsDATA(muNofMuonHits);
   muNofMuonHits->Write();
   delete muNofMuonHits;

  TH1F * muChi2= new TH1F("muChi2", "muChi2", 100, 0, 100);
 chainDATA.Project("muChi2", "MuChi2", stdCut);
 setGraphicsDATA(muChi2);
   muChi2->Write();
   delete muChi2;

out.cd("/");



//==============   histo MC ===============///
TDirectory * dir = out.mkdir("MuMCPlots");
  dir->cd();

 TH1F * muPtMCDen= new TH1F("muPtMCDen", "muPtMCDen",  100, 0, 100);
 chainMC.Project("muPtMCDen", "MuPt", qualCut);
 setGraphicsMC(muPtMCDen);
   muPtMCDen->Write();
   delete muPtMCDen;

TH1F * muEtaMC= new TH1F("muEtaMC", "muEtaMC",  100, -2.5, 2.5);
 chainMC.Project("muEtaMC", "MuEta", qualCut);
 setGraphicsMC(muEtaMC);
   muEtaMC->Write();
   delete muEtaMC;

 TH1F * muPtMCNum= new TH1F("muPtMCNum", "muPtMCNum",  100, 0, 100);
 chainMC.Project("muPtMCNum", "MuPt", qualCutHLT);
 setGraphicsMC(muPtMCNum);
   muPtMCNum->Write();
   delete muPtMCNum;


   TH2F * muPtEtaMCDen= new TH2F("muPtEtaMCDen", "muPtEtaMCDenEta",   100, -2.5, 2.5, 100, 0, 100);
 chainMC.Project("muPtEtaMCDen", "MuPt:MuEta", qualCut);
 //setGraphics2dimMC(muPtEtaMCDen);
   muPtEtaMCDen->Write();
   delete muPtEtaMCDen;


   TH2F * muPtEtaMCNum= new TH2F("muPtEtaMCNum", "muPtEtaMCNum",  100, -2.5, 2.5,  100, 0, 100);
 chainMC.Project("muPtEtaMCNum", "MuPt:MuEta", qualCutHLT);
 // setGraphics2dimMC(muPtEtaMCNum);
   muPtEtaMCNum->Write();
   delete muPtEtaMCNum;



   TH1F * muNoHLTEta= new TH1F("muNoHLTEta", "muNoHLTEta", 100, -2.5, 2.5);
 chainMC.Project("muNoHLTEta", "MuEta", qualCutNoHLT);
 setGraphicsMC(muNoHLTEta);
   muNoHLTEta->Write();
   delete muNoHLTEta;

   TH1F * muNoHLTPixelHits= new TH1F("muNoHLTPixelHits", "muNoHLTPixelHits", 10, -0.5, 9.5);
 chainMC.Project("muNoHLTPixelHits", "MuNofPixelHits", qualCutNoHLT);
 setGraphicsMC(muNoHLTPixelHits);
 muNoHLTPixelHits->Write();
   delete muNoHLTPixelHits;

  TH1F * muNofPixelHits= new TH1F("muNofPixelHits", "muNofPixelHits", 10, -0.5, 9.5);
 chainMC.Project("muNofPixelHits", "MuNofPixelHits", stdCut);
 setGraphicsMC(muNofPixelHits);
   muNofPixelHits->Write();
   delete muNofPixelHits;

  TH1F * muNofStripHits= new TH1F("muNofStripHits", "muNofStripHits", 30, -0.5, 29.5);
 chainMC.Project("muNofStripHits", "MuNofStripHits", stdCut);
 setGraphicsMC(muNofStripHits);
   muNofStripHits->Write();
   delete muNofStripHits;


  TH1F * muNofMuonHits= new TH1F("muNofMuonHits", "muNofMuonHits", 40, -0.5, 39.5);
 chainMC.Project("muNofMuonHits", "MuNofMuonHits", stdCut);
 setGraphicsMC(muNofMuonHits);
   muNofMuonHits->Write();
   delete muNofMuonHits;

  TH1F * muChi2= new TH1F("muChi2", "muChi2", 100, 0, 100);
 chainMC.Project("muChi2", "MuChi2", stdCut);
 setGraphicsMC(muChi2);
   muChi2->Write();
   delete muChi2;

out.cd("/");




//==============   histo DATA vs MC ===============///
TDirectory * dir = out.mkdir("DATAvsMCPlots");
  dir->cd();

  TH1F * muPtDATADen= (TH1F*) out.Get("MuDATAPlots/muPtDATADen");
  TH1F * muPtMCDen= (TH1F*) out.Get("MuMCPlots/muPtMCDen");  
  compareDATAMC( "PtDen", muPtDATADen, muPtMCDen );
 
  TH1F * muPtDATANum= (TH1F*) out.Get("MuDATAPlots/muPtDATANum");
  TH1F * muPtMCNum= (TH1F*) out.Get("MuMCPlots/muPtMCNum");  
  compareDATAMC( "PtNum", muPtDATANum, muPtMCNum );
 
  
  TH1F * muEtaDATA= (TH1F*) out.Get("MuDATAPlots/muEtaDATA");
  TH1F * muEtaMC= (TH1F*) out.Get("MuMCPlots/muEtaMC");  
  compareDATAMC( "Eta", muEtaDATA, muEtaMC );

   TH1F * muNoHLTEtaDATA= (TH1F*) out.Get("MuDATAPlots/muNoHLTEta");
  TH1F * muNoHLTEtaMC= (TH1F*) out.Get("MuMCPlots/muNoHLTEta");  
  compareDATAMC( "NoHLTEta", muNoHLTEtaDATA, muNoHLTEtaMC );


   TH1F * muNoHLTPixelHitsDATA= (TH1F*) out.Get("MuDATAPlots/muNoHLTPixelHits");
  TH1F * muNoHLTPixelHitsMC= (TH1F*) out.Get("MuMCPlots/muNoHLTPixelHits");  
  compareDATAMC( "NoHLTPixelHits", muNoHLTPixelHitsDATA, muNoHLTPixelHitsMC );


   TH1F * muPixelHitsDATA= (TH1F*) out.Get("MuDATAPlots/muNofPixelHits");
  TH1F * muPixelHitsMC= (TH1F*) out.Get("MuMCPlots/muNofPixelHits");  
  compareDATAMC( "PixelHits", muPixelHitsDATA, muPixelHitsMC );



   TH1F * muStripHitsDATA= (TH1F*) out.Get("MuDATAPlots/muNofStripHits");
  TH1F * muStripHitsMC= (TH1F*) out.Get("MuMCPlots/muNofStripHits");  
  compareDATAMC( "StripHits", muStripHitsDATA, muStripHitsMC );


   TH1F * muMuonHitsDATA= (TH1F*) out.Get("MuDATAPlots/muNofMuonHits");
  TH1F * muMuonHitsMC= (TH1F*) out.Get("MuMCPlots/muNofMuonHits");  
  compareDATAMC( "MuonHits", muMuonHitsDATA, muMuonHitsMC );

   TH1F * muChi2DATA= (TH1F*) out.Get("MuDATAPlots/muChi2");
  TH1F * muChi2MC= (TH1F*) out.Get("MuMCPlots/muChi2");  
  compareDATAMC( "Chi2", muChi2DATA, muChi2MC );





out.cd("/");








  

 out.Close();


}
