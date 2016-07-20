#include "TMath.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TLegend.h"
#include "TLatex.h"
#include "TMultiGraph.h"

#include <iostream>

#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1UpgradeDataFormat.h"

TH1F * hset(TH1F * h, int color){
  h->SetLineWidth(2);
  h->SetLineColor(color);
  h->SetMarkerStyle(23);
  h->SetMarkerColor(color);
  return h;
}

TH1F * norm(TH1F * h){
  double n = h->GetEntries();
  if (n>0.0){
    h->Scale(1.0/n);
  }
  h->SetMaximum(1.0);
  h->SetMinimum(0);
  return h;
}

void kin(const char * rootfile="L1Ntuple.root",const char * treepath="l1UpgradeEmuTree/L1UpgradeTree"){
  gStyle->SetOptStat(0);

  // make trees
  TFile * file = new TFile(rootfile);
  TTree * treeL1Up  = (TTree*) file->Get(treepath);
  if (! treeL1Up){
    cout << "ERROR: could not open tree\n";
    return;
  }
  treeL1Up->Print();

  // set branch addresses
  //L1Analysis::L1AnalysisL1UpgradeDataFormat    *upgrade_ = new L1Analysis::L1AnalysisL1UpgradeDataFormat();
  //treeL1Up->SetBranchAddress("L1Upgrade", &upgrade_);

  TH1F* muEt   = hset(new TH1F("muEt", "", 50, 0.0, 40.0), kBlue);
  TH1F* muPhi  = hset(new TH1F("muPhi", "", 50, -3.0, 3.0), kBlue);
  TH1F* muEta  = hset(new TH1F("muEta", "", 50, -3.0, 3.0), kBlue);

  TH1F* jetEt  = hset(new TH1F("jetEt", "", 50, 0.0, 40.0), kRed);
  TH1F* jetPhi = hset(new TH1F("jetPhi", "", 50, -3.0, 3.0), kRed);
  TH1F* jetEta = hset(new TH1F("jetEta", "", 50, -3.0, 3.0), kRed);

  TH1F* egEt   = hset(new TH1F("egEt", "", 50, 0.0, 40.0), kGreen);
  TH1F* egPhi  = hset(new TH1F("egPhi", "", 50, -3.0, 3.0), kGreen);
  TH1F* egEta  = hset(new TH1F("egEta", "", 50, -3.0, 3.0), kGreen);

  TH1F* tauEt  = hset(new TH1F("tauEt", "", 50, 0.0, 40.0), kOrange);
  TH1F* tauPhi = hset(new TH1F("tauPhi", "", 50, -3.0, 3.0), kOrange);
  TH1F* tauEta = hset(new TH1F("tauEta", "", 50, -3.0, 3.0), kOrange);

  TCanvas * c0 = new TCanvas();
  treeL1Up->Draw("muonEt>>muEt", "muonQual>=8");
  treeL1Up->Draw("muonPhi>>muPhi", "muonQual>=8");
  treeL1Up->Draw("muonEta>>muEta", "muonQual>=8");
  treeL1Up->Draw("jetEt>>jetEt");
  treeL1Up->Draw("jetPhi>>jetPhi");
  treeL1Up->Draw("jetEta>>jetEta");
  treeL1Up->Draw("egEt>>egEt");
  treeL1Up->Draw("egPhi>>egPhi");
  treeL1Up->Draw("egEta>>egEta");
  treeL1Up->Draw("tauEt>>tauEt");
  treeL1Up->Draw("tauPhi>>tauPhi");
  treeL1Up->Draw("tauEta>>tauEta");
  delete c0;

  norm(muEt);    norm(jetEt);    norm(tauEt);    norm(egEt); 
  norm(muPhi);   norm(jetPhi);   norm(tauPhi);   norm(egPhi); 
  norm(muEta);   norm(jetEta);   norm(tauEta);   norm(egEta); 

  TCanvas* c1 = new TCanvas;
  muEt->SetMaximum(0.4);
  muEt->SetXTitle("Et [GeV]");
  muEt->Draw("H");
  jetEt->Draw("HSAME");
  tauEt->Draw("HSAME");
  egEt->Draw("HSAME");

  TLegend* leg1 = new TLegend(0.4,0.5,0.6,0.9);
  leg1->SetFillColor(0);
  leg1->AddEntry(egEt,"EGamma","lp");
  leg1->AddEntry(tauEt,"Tau","lp");
  leg1->AddEntry(jetEt,"Jets","lp");
  leg1->AddEntry(muEt,"Muons","lp");
  leg1->SetBorderSize(0);
  leg1->SetFillStyle(0);
  leg1->Draw();
  c1->SaveAs("et.pdf");

  TCanvas* c2 = new TCanvas;
  muEta->SetMaximum(0.2);
  muEta->SetXTitle("Eta");
  muEta->Draw("H");
  jetEta->Draw("HSAME");
  tauEta->Draw("HSAME");
  egEta->Draw("HSAME");
  leg1->Draw();
  c2->SaveAs("eta.pdf");

  TCanvas* c3 = new TCanvas;
  muPhi->SetXTitle("Phi");
  muPhi->SetMaximum(0.07);
  muPhi->Draw("H");
  jetPhi->Draw("HSAME");
  tauPhi->Draw("HSAME");
  egPhi->Draw("HSAME");

  TLegend* leg3 = new TLegend(0.2,0.5,0.4,0.9);
  leg3->SetFillColor(0);
  leg3->AddEntry(egEt,"EGamma","lp");
  leg3->AddEntry(tauEt,"Tau","lp");
  leg3->AddEntry(jetEt,"Jets","lp");
  leg3->AddEntry(muEt,"Muons","lp");
  leg3->SetBorderSize(0);
  leg3->SetFillStyle(0);
  leg3->Draw();
  c3->SaveAs("phi.pdf");


}
