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
#include "plotHist.cc"
#include <string>
#include <map>



void matchcalculator(){


  gROOT->Reset();
  
  gROOT->SetStyle("Plain");

  gStyle->SetCanvasColor(kWhite);

  gStyle->SetCanvasBorderMode(0);     // turn off canvas borders
  gStyle->SetPadBorderMode(0);
  gStyle->SetOptStat(1111);
  gStyle->SetOptTitle(1);
  
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
  
  
  int ncut1=72;
  int ncut2=108;


  TCanvas* c1 = new TCanvas("c1","Track performance",200,10,700,800);
  c1->Divide(2,2);
  c1->SetFillColor(0);
  c1->SetGrid();
  
  double max=128.0;
 
  TH1 *hist1 = new TH1F("h1","MatchesAll Layer",max,0.5,max+0.5);
  TH1 *hist2 = new TH1F("h2","MatchesPass Layer",max,0.5,max+0.5);

  TH1 *hist3 = new TH1F("h3","MatchesAll Disk",max,0.5,max+0.5);
  TH1 *hist4 = new TH1F("h4","MatchesPass Disk",max,0.5,max+0.5);
  
  ifstream in("matchcalculator.txt");

  int count=0;

  std::map<TString, std::pair<TH1*,TH1*> > hists;

  while (in.good()){

    TString name;
    int matchesAll,matchesPass;
  
    in >>name>>matchesAll>>matchesPass;

    if (!in.good()) continue;
    //cout << name <<" "<< countall <<" "<< countpass << endl;

    if (matchesAll>max) matchesAll=max;
    if (matchesPass>max) matchesPass=max;

    if (name[3]=='L'){
      hist1->Fill(matchesAll);
      hist2->Fill(matchesPass);
    }
    
    if (name[3]=='D'){
      hist3->Fill(matchesAll);
      hist4->Fill(matchesPass);
    }

    std::map<TString, std::pair<TH1*,TH1*> >::iterator it=hists.find(name);
   
    if (it==hists.end()) {
      TString name1=name+"_All";
      TH1 *histAll = new TH1F(name1,name1,max,+0.5,max+0.5);
      histAll->Fill(matchesAll);
      name1=name+"_Pass";
      TH1 *histPass = new TH1F(name1,name1,max,+0.5,max+0.5);
      histPass->Fill(matchesPass);
      std::pair<TH1*,TH1*> histpair(histAll,histPass);
      hists[name]=histpair;
    } else {
      hists[name].first->Fill(matchesAll);
      hists[name].second->Fill(matchesPass);
   }

    count++;

  }

  cout << "count = "<<count<<endl;

  c1->cd(1);
  plotHist(hist1,0.05,ncut1,ncut2);
  hist2->SetLineColor(kBlue);
  hist2->Draw("same");

  c1->cd(2);
  plotHist(hist3,0.05,ncut1,ncut2);
  hist4->SetLineColor(kBlue);
  hist4->Draw("same");

  c1->Print("matchcalculator.pdf(","pdf");

  int pages=0;

  std::map<TString, std::pair<TH1*,TH1*> >::iterator it=hists.begin();

  TCanvas* c=0;

  while(it!=hists.end()) {

    if (pages%4==0) {
     
      c = new TCanvas(it->first,"Track performance",200,50,600,700);
      c->Divide(2,2);
      c->SetFillColor(0);
      c->SetGrid();

    }

    c->cd(pages%4+1);
    //gPad->SetLogy();
    plotHist(it->second.first,0.05,ncut1,ncut2);
    it->second.second->SetLineColor(kBlue);
    it->second.second->Draw("same");
    
    pages++;

    if (pages%4==0) {
      c->Print("matchcalculator.pdf","pdf");
    }

    ++it;


 }

  c->Print("matchcalculator.pdf)","pdf");

}
