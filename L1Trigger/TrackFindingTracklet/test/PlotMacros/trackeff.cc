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


void trackeff(){

//
// To see the output of this macro, click here.

//


  gROOT->Reset();
  gROOT->SetStyle("Plain");
  
  gStyle->SetCanvasColor(kWhite);
  
  gStyle->SetCanvasBorderMode(0);     // turn off canvas borders
  gStyle->SetPadBorderMode(0);
  gStyle->SetOptStat(0);


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
  //c1_4->SetGridy(10);
  //c1_3->SetGridy(10);
  
  TString parttype="Muon";
  //TString parttype="Electron";
  //TString parttype="Pion";
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
  
  double effmin=0.;
  int nbin=32;
  double maxpt=10.0;
  
  double NSector=9.0;
  double twopi=8*atan(1.0);
  double sectorphi=twopi/NSector;
  
  
  TH1 *hist1 = new TH1F("h1","Generated Sim Tracks",nbin,2.0,maxpt);
  TH1 *hist2 = new TH1F("h2",parttype+" track finding",nbin,2.0,maxpt);
  
  TH1 *hist3 = new TH1F("h3","Generated Sim Tracks",50,-2.5,2.5);
  TH1 *hist4 = new TH1F("h4",parttype+" track finding",50,-2.5,2.5);
  
  TH1 *hist5 = new TH1F("h5","Generated Sim Tracks",50,-3.1415,3.1415);
  TH1 *hist6 = new TH1F("h6",parttype+" track finding",50,-3.1415,3.1415);
  
  TH1 *hist7 = new TH1F("h7","Generated Sim Tracks",50,0.0,sectorphi);
  TH1 *hist8 = new TH1F("h8",parttype+" track finding",50,0.0,sectorphi);
  
  hist1->GetXaxis()->SetTitle("pT (GeV)");
  hist1->GetYaxis()->SetTitle("Events");
  
  hist2->GetXaxis()->SetTitle("pT (GeV)");
  hist2->GetYaxis()->SetTitle("Efficiency");
  
  hist3->GetXaxis()->SetTitle("#eta");
  hist3->GetYaxis()->SetTitle("Events");
  
  hist4->GetXaxis()->SetTitle("#eta");
  hist4->GetYaxis()->SetTitle("Efficiency");
  
  hist5->GetXaxis()->SetTitle("#phi");
  hist5->GetYaxis()->SetTitle("Events");
  
  hist6->GetXaxis()->SetTitle("#phi");
  hist6->GetYaxis()->SetTitle("Efficiency");
  
  hist7->GetXaxis()->SetTitle("#phi");
  hist7->GetYaxis()->SetTitle("Events");
  
  hist8->GetXaxis()->SetTitle("#phi");
  hist8->GetYaxis()->SetTitle("Efficiency");
  
  
  ifstream in("trackeff.txt");
  
  int count=0;
  int counteff=0;
  
  
  
  while (in.good()){
    
    count++;
    
    double pt,eta,phi0,eff;
    
    in>>pt>>eta>>phi0>>eff;
    
    double phisector=phi0;
    
    while(phisector<0.0) phisector+=sectorphi;
    while(phisector>sectorphi) phisector-=sectorphi;
    
    hist7->Fill(phisector);
    hist5->Fill(phi0);
    hist3->Fill(eta);
    hist1->Fill(pt);
    if (eff>0.5){
      counteff++;
      hist2->Fill(pt);
      hist4->Fill(eta);
      hist6->Fill(phi0);
      hist8->Fill(phisector);
    }
    
  }
  
  cout << "Processed: "<<count<<" events with <eff>="<<counteff*1.0/count<<endl;
  
  //c1->cd(1);
  //h1->Draw();
  //h2->Draw("SAME");
  
  hist2->Divide(hist1);
  hist6->Divide(hist5);
  hist8->Divide(hist7);
  hist4->Divide(hist3);
  
  c1->cd(2);
  hist2->SetMinimum(1.05);
  hist2->SetMinimum(effmin);
  hist2->Draw();
  
  c1->cd(1);
  hist6->SetMaximum(1.05);
  hist6->SetMinimum(effmin);
  hist6->DrawCopy();

  c1->cd(3);
  hist4->SetMaximum(1.05);
  hist4->SetMinimum(effmin);
  hist4->DrawCopy();
  
  c1->cd(4);
  hist8->SetMaximum(1.05);
  hist8->SetMinimum(effmin);
  hist8->Draw();
  
  c1->Print(parttype+"_trackeff.pdf");
  c1->Print(parttype+"_trackeff.png");
 

}


