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
#include <string>
#include <map>
#include "plotHist.cc"



void trackletcalculator(){
  //
  // To see the output of this macro, click here.
  //

  int ncut1=72;
  int ncut2=108;
  
  gROOT->Reset();

  gROOT->SetStyle("Plain");

  gStyle->SetCanvasColor(kWhite);

  gStyle->SetCanvasBorderMode(0);     // turn off canvas borders
  gStyle->SetPadBorderMode(0);
  gStyle->SetOptStat(1111);
  gStyle->SetOptTitle(1);

  // For publishing:
  gStyle->SetLineWidth(2);
  gStyle->SetTextSize(1.1);
  gStyle->SetLabelSize(0.06,"xy");
  gStyle->SetTitleSize(0.06,"xy");
  gStyle->SetTitleOffset(1.2,"x");
  gStyle->SetTitleOffset(1.0,"y");
  gStyle->SetPadTopMargin(0.1);
  gStyle->SetPadRightMargin(0.1);
  gStyle->SetPadBottomMargin(0.16);
  gStyle->SetPadLeftMargin(0.12);




  TCanvas* c1 = new TCanvas("c1","Track performance",200,10,700,800);
  c1->Divide(2,2);
  c1->SetFillColor(0);
  c1->SetGrid();

  TCanvas* c2 = new TCanvas("c2","Track performance",200,10,700,800);
  c2->Divide(2,2);
  c2->SetFillColor(0);
  c2->SetGrid();

  TCanvas* c3 = new TCanvas("c3","Track performance",200,10,700,800);
  c3->Divide(2,2);
  c3->SetFillColor(0);
  c3->SetGrid();
  
  double max=128.0;
 
  TH1 *hist1 = new TH1F("h1","TrackletsAll Layer",max+1,-0.5,max+0.5);
  TH1 *hist2 = new TH1F("h2","TrackletsPass Layer",max+1,-0.5,max+0.5);

  TH1 *hist3 = new TH1F("h3","TrackletsAll Disk",max+1,-0.5,max+0.5);
  TH1 *hist4 = new TH1F("h4","TrackletsPass Disk",max+1,-0.5,max+0.5);

  TH1 *hist10 = new TH1F("h10","TrackletsAll L1L2",max+1,-0.5,max+0.5);
  TH1 *hist20 = new TH1F("h20","TrackletsPass L1L2",max+1,-0.5,max+0.5);

  TH1 *hist11 = new TH1F("h11","TrackletsAll L3L4",max+1,-0.5,max+0.5);
  TH1 *hist21 = new TH1F("h21","TrackletsPass L3L4",max+1,-0.5,max+0.5);

  TH1 *hist12 = new TH1F("h12","TrackletsAll L5L6",max+1,-0.5,max+0.5);
  TH1 *hist22 = new TH1F("h22","TrackletsPass L5L6",max+1,-0.5,max+0.5);

  TH1 *hist13 = new TH1F("h13","TrackletsAll D1D2",max+1,-0.5,max+0.5);
  TH1 *hist23 = new TH1F("h23","TrackletsPass D1D2",max+1,-0.5,max+0.5);

  TH1 *hist14 = new TH1F("h14","TrackletsAll D3D4",max+1,-0.5,max+0.5);
  TH1 *hist24 = new TH1F("h24","TrackletsPass D3D4",max+1,-0.5,max+0.5);

  TH1 *hist15 = new TH1F("h15","TrackletsAll L1D1",max+1,-0.5,max+0.5);
  TH1 *hist25 = new TH1F("h25","TrackletsPass L1D1",max+1,-0.5,max+0.5);
  
  TH1 *hist16 = new TH1F("h16","TrackletsAll L2D1",max+1,-0.5,max+0.5);
  TH1 *hist26 = new TH1F("h26","TrackletsPass L2D1",max+1,-0.5,max+0.5);
  
  
  ifstream in("trackletcalculator.txt");

  int count=0;

  std::map<TString, std::pair<TH1*,TH1*> > hists;

  while (in.good()){

    TString name;
    int trackletsAll,trackletsPass;
    
    in >>name>>trackletsAll>>trackletsPass;

    if (!in.good()) continue;
    //cout << name <<" "<< countall <<" "<< countpass << endl;

    if (trackletsAll>max) trackletsAll=max;
    if (trackletsPass>max) trackletsPass=max;

    if (name.Contains("TC_L")){
      hist1->Fill(trackletsAll);
      hist2->Fill(trackletsPass);
    }

    if (name.Contains("TC_L1L2")){
      hist10->Fill(trackletsAll);
      hist20->Fill(trackletsPass);
    }

    if (name.Contains("TC_L3L4")){
      hist11->Fill(trackletsAll);
      hist21->Fill(trackletsPass);
    }

    if (name.Contains("TC_L5L6")){
      hist12->Fill(trackletsAll);
      hist22->Fill(trackletsPass);
    }

    if (name.Contains("TC_D1D2")){
      hist13->Fill(trackletsAll);
      hist23->Fill(trackletsPass);
    }

    if (name.Contains("TC_D3D4")){
      hist14->Fill(trackletsAll);
      hist24->Fill(trackletsPass);
    }
    
    if (name.Contains("TC_L1D1")){
      hist15->Fill(trackletsAll);
      hist25->Fill(trackletsPass);
    }
    
    if (name.Contains("TC_L2D1")){
      hist16->Fill(trackletsAll);
      hist26->Fill(trackletsPass);
    }
    

    if (name.Contains("TC_F")||
	name.Contains("TC_B")||
	name.Contains("TC_D")){
      hist3->Fill(trackletsAll);
      hist4->Fill(trackletsPass);
    }
    
    std::map<TString, std::pair<TH1*,TH1*> >::iterator it=hists.find(name);
    
    if (it==hists.end()) {
      TString name1=name+"_All";
      TH1 *histAll = new TH1F(name1,name1,max+1,-0.5,max+0.5);
      histAll->Fill(trackletsAll);
      name1=name+"_Pass";
      TH1 *histPass = new TH1F(name1,name1,max+1,-0.5,max+0.5);
      histPass->Fill(trackletsPass);
      std::pair<TH1*,TH1*> histpair(histAll,histPass);
      hists[name]=histpair;
    } else {
      hists[name].first->Fill(trackletsAll);
      hists[name].second->Fill(trackletsPass);
    }

    count++;

  }

  cout << "count = "<<count<<endl;

  c1->cd(1);
  plotHist(hist1,0.05,ncut1,ncut2);
  hist2->SetLineColor(kBlue);
  hist2->Draw("same");

  c1->cd(2);
  gPad->SetLogy();
  plotHist(hist1,0.05,ncut1,ncut2);
  hist2->SetLineColor(kBlue);
  hist2->Draw("same");


  c1->cd(3);
  plotHist(hist3,0.05,ncut1,ncut2);
  hist4->SetLineColor(kBlue);
  hist4->Draw("same");
  
  c1->cd(4);
  gPad->SetLogy();
  plotHist(hist3,0.05,ncut1,ncut2);
  hist4->SetLineColor(kBlue);
  hist4->Draw("same");
  
  c1->Print("trackletcalculator.pdf(","pdf");

  c2->cd(1);
  gPad->SetLogy();
  plotHist(hist10,0.05,ncut1,ncut2,true);
  hist20->SetLineColor(kBlue);
  hist20->Draw("same");

  c2->cd(2);
  gPad->SetLogy();
  plotHist(hist11,0.05,ncut1,ncut2,true);
  hist21->SetLineColor(kBlue);
  hist21->Draw("same");

  c2->cd(3);
  gPad->SetLogy();
  plotHist(hist12,0.05,ncut1,ncut2,true);
  hist22->SetLineColor(kBlue);
  hist22->Draw("same");


  c2->Print("trackletcalculator.pdf","pdf");

  c3->cd(1);
  gPad->SetLogy();
  plotHist(hist13,0.05,ncut1,ncut2,true);
  hist23->SetLineColor(kBlue);
  hist23->Draw("same");


  c3->cd(2);
  gPad->SetLogy();
  plotHist(hist14,0.05,ncut1,ncut2,true);
  hist24->SetLineColor(kBlue);
  hist24->Draw("same");

  c3->cd(3);
  gPad->SetLogy();
  plotHist(hist15,0.05,ncut1,ncut2,true);
  hist25->SetLineColor(kBlue);
  hist25->Draw("same");

  c3->cd(4);
  gPad->SetLogy();
  plotHist(hist16,0.05,ncut1,ncut2,true);
  hist26->SetLineColor(kBlue);
  hist26->Draw("same");




  
  c3->Print("trackletcalculator.pdf","pdf");

  
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
    gPad->SetLogy();
    it->second.second->SetLineColor(kBlue);
    plotHist(it->second.second,0.05,ncut1,ncut2);
    it->second.first->SetLineColor(kRed);
    it->second.first->Draw("same");
    
    
    pages++;

	++it;

	if (it==hists.end()) {
	  c->Print("trackletcalculator.pdf)","pdf");
	}
	else {
      if (pages%4==0) {
        c->Print("trackletcalculator.pdf","pdf");
      }
	}

   
    //if (it==hists.end()&&(pages%4!=0)) {
    //  c->Print("trackletcalculator.pdf)","pdf");
    //}
    
  }

  //c->Print("trackletcalculator.pdf)","pdf");


  
}
