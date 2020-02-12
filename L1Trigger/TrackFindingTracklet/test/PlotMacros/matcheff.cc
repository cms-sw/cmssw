#include "TMath.h"
#include "TRint.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TLorentzVector.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TGaxis.h"
#include <fstream>
#include <iostream>
#include "TMath.h"


void matcheff(){

//
// To see the output of this macro, click here.

//

//#include "TMath.h"

  gROOT->Reset();
  gROOT->SetStyle("Plain");
  
  gStyle->SetCanvasColor(kWhite);

  gStyle->SetCanvasBorderMode(0);     // turn off canvas borders
  gStyle->SetPadBorderMode(0);
  //gStyle->SetOptStat(0);


  // For publishing:
  gStyle->SetLineWidth(1);
  gStyle->SetTextSize(1.1);
  gStyle->SetLabelSize(0.06,"xy");
  gStyle->SetTitleSize(0.06,"xy");
  gStyle->SetTitleOffset(1.2,"x");
  gStyle->SetTitleOffset(1.0,"y");
  gStyle->SetPadTopMargin(0.1);
  gStyle->SetPadRightMargin(0.1);
  gStyle->SetPadBottomMargin(0.16);
  gStyle->SetPadLeftMargin(0.12);


  TCanvas* c1 = new TCanvas("c1","Track performance",200,10,1000,1100);
  c1->Divide(2,2);
  c1->SetFillColor(0);
  
  TCanvas* c2 = new TCanvas("c2","Track performance",200,10,1000,1100);
  c2->Divide(2,2);
  c2->SetFillColor(0);
  
  //c1_4->SetGridy(10);
  //c1_3->SetGridy(10);
  
  //TString parttype="All Tracks";
  //TString parttype="Electron";
  TString parttype="Pion";
  //TString parttype="Kaon";
  //TString parttype="Proton";

  int itype=-1;

  if (parttype=="Muon") itype=13;
  if (parttype=="Electron") itype=11;
  if (parttype=="Pion") itype=211;
  if (parttype=="Kaon") itype=321;
  if (parttype=="Proton") itype=2212;
  
  if (itype==-1) {
    cout << "Need to correctly specify parttype:"<<parttype<<endl;
  }
  
  double effmin=0.0;
  int nbin=32;
  double maxpt=20.0;
  
  double NSector=9.0;
  double twopi=8*atan(1.0);
  double sectorphi=twopi/NSector;
   
 
  TH1 *hist1 = new TH1F("h1","Generated Sim Tracks",nbin,2.0,maxpt);
 TH1 *hist2 = new TH1F("h2",parttype+" track finding",nbin,2.0,maxpt);
  TH1 *hist12 = new TH1F("h12",parttype+" track finding",nbin,2.0,maxpt);

  TH1 *hist3 = new TH1F("h3","Generated Sim Tracks",50,-2.5,2.5);
  TH1 *hist4 = new TH1F("h4",parttype+" track finding",50,-2.5,2.5);
  TH1 *hist14 = new TH1F("h14",parttype+" track finding",50,-2.5,2.5);

  TH1 *hist5 = new TH1F("h5","Generated Sim Tracks",50,-3.1415,3.1415);
  TH1 *hist6 = new TH1F("h6",parttype+" track finding",50,-3.1415,3.1415);
  TH1 *hist16 = new TH1F("h16",parttype+" track finding",50,-3.1415,3.1415);

  TH1 *hist7 = new TH1F("h7","Generated Sim Tracks",50,0.0,sectorphi);
  TH1 *hist8 = new TH1F("h8",parttype+" track finding",50,0.0,sectorphi);
  TH1 *hist18 = new TH1F("h18",parttype+" track finding",50,0.0,sectorphi);

  TH1 *hist21 = new TH1F("h21","pT Res",50,-10.0,10.0);
  hist21->GetXaxis()->SetTitle("#sigma(p_{T})/p_{T} [%]");
  TH1 *hist22 = new TH1F("h22","phi Res",50,-10.0,10.0);
  hist22->GetXaxis()->SetTitle("#sigma(#phi_{0}) [mrad]");
  TH1 *hist23 = new TH1F("h23","eta Res",50,-0.05,0.05);
  hist23->GetXaxis()->SetTitle("#sigma(#eta_{0})");
  TH1 *hist24 = new TH1F("h24","z0 Res",50,-2.0,2.0);
  hist24->GetXaxis()->SetTitle("#sigma(z_{0}) [cm]");

  
  hist1->GetXaxis()->SetTitle("pT (GeV)");
  hist1->GetYaxis()->SetTitle("Events");

  hist2->GetXaxis()->SetTitle("pT (GeV)");
  hist2->GetYaxis()->SetTitle("Efficiency");

  hist12->GetXaxis()->SetTitle("pT (GeV)");
  hist12->GetYaxis()->SetTitle("Efficiency");

  hist3->GetXaxis()->SetTitle("#eta");
  hist3->GetYaxis()->SetTitle("Events");

  hist4->GetXaxis()->SetTitle("#eta");
  hist4->GetYaxis()->SetTitle("Efficiency");

  hist14->GetXaxis()->SetTitle("#eta");
  hist14->GetYaxis()->SetTitle("Efficiency");

  hist5->GetXaxis()->SetTitle("#phi");
  hist5->GetYaxis()->SetTitle("Events");

  hist6->GetXaxis()->SetTitle("#phi");
  hist6->GetYaxis()->SetTitle("Efficiency");

  hist16->GetXaxis()->SetTitle("#phi");
  hist16->GetYaxis()->SetTitle("Efficiency");

  hist7->GetXaxis()->SetTitle("#phi");
  hist7->GetYaxis()->SetTitle("Events");

  hist8->GetXaxis()->SetTitle("#phi");
  hist8->GetYaxis()->SetTitle("Efficiency");

  hist18->GetXaxis()->SetTitle("#phi");
  hist18->GetYaxis()->SetTitle("Efficiency");


  ifstream in("matcheff.txt");

  int count=0;
  int counteff=0;
  int counteffloose=0;
 
  
  while (in.good()){
   
    int event,simeventid,simtrackid,type,eff,effloose;
    
    double pt,eta,phi0,vx,vy,vz,dpt,dphi,deta,dz0;
    
    in>>event>>simeventid>>simtrackid>>type>>pt>>eta>>phi0>>vx>>vy>>vz>>eff>>effloose
      >>dpt>>dphi>>deta>>dz0;
    
    //if (eta>1.0) continue;
    if (pt<2.0) continue;
    //if (pt>3.0) continue;
    //if (type>0.0) continue;
    //if (pt>3.0) continue;
    //if (pt>10) continue;
    
    //if (simeventid>0) continue;
    
    //if (fabs(eta)<1.8) continue;
    
    if (abs(type)!=itype) continue;
    
    count++;
    
    double phisector=phi0;
    
    while(phisector<0.0) phisector+=sectorphi;
    while(phisector>sectorphi) phisector-=sectorphi;
    
    hist7->Fill(phisector);
    hist5->Fill(phi0);
    hist3->Fill(eta);
    hist1->Fill(pt);
    if (eff==1){
      counteff++;
      hist2->Fill(pt);
      hist4->Fill(eta);
      hist6->Fill(phi0);
      hist8->Fill(phisector);
    }
    if (effloose==1){
      counteffloose++;
      hist12->Fill(pt);
      hist14->Fill(eta);
      hist16->Fill(phi0);
      hist18->Fill(phisector);
    }

    if (eff) {
    //if (effloose&!eff) { // <= to check those that pass loose but fails tight matching
      hist21->Fill(100*dpt/pt);
      hist22->Fill(dphi*1000);
      hist23->Fill(deta);
      if (simeventid==0) hist24->Fill(dz0);
    }
    
  }
  
  cout << "Processed: "<<count<<" events with <eff>="<<counteff*1.0/count
       <<" and <eff loose>="<<counteffloose*1.0/count<<endl;
  
  
  hist2->Sumw2();
  hist6->Sumw2();
  hist8->Sumw2();
  hist4->Sumw2();
  
  hist2->Divide(hist2,hist1,1,1,"B");
  hist6->Divide(hist6,hist5,1,1,"B");
  hist8->Divide(hist8,hist7,1,1,"B");
  hist4->Divide(hist4,hist3,1,1,"B");
  
  hist12->Sumw2();
  hist16->Sumw2();
  hist18->Sumw2();
  hist14->Sumw2();
  
  hist12->Divide(hist12,hist1,1,1,"B");
  hist16->Divide(hist16,hist5,1,1,"B");
  hist18->Divide(hist18,hist7,1,1,"B");
  hist14->Divide(hist14,hist3,1,1,"B");
 

  c1->cd(2);
  hist2->SetMinimum(1.05);
  hist2->SetMinimum(effmin);
  hist2->Draw();
  hist12->SetLineColor(kBlue);
  hist12->Draw("same");
  
  

  c1->cd(1);
  hist6->SetMaximum(1.05);
  hist6->SetMinimum(effmin);
  hist6->DrawCopy();
  hist16->SetLineColor(kBlue);
  hist16->Draw("same");
  
  
  c1->cd(3);
  hist4->SetMaximum(1.05);
  hist4->SetMinimum(effmin);
  hist4->DrawCopy();
  hist14->SetLineColor(kBlue);
  hist14->Draw("same");
  
  
  c1->cd(4);
  hist8->SetMaximum(1.05);
  hist8->SetMinimum(effmin);
  hist8->Draw();
  hist18->SetLineColor(kBlue);
  hist18->Draw("same");
  


  c1->Print(parttype+"_trackeff.pdf");
  c1->Print(parttype+"_trackeff.png");

  c2->cd(1);
  hist21->Draw();
  c2->cd(2);
  hist22->Draw();
  c2->cd(3);
  hist23->Draw();
  c2->cd(4);
  hist24->Draw();

  c2->Print(parttype+"_resolution.pdf");

  
}


