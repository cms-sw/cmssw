#include "TMath.h"
#include "TRint.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TLorentzVector.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TF1.h"
#include "TGaxis.h"
#include <fstream>
#include <iostream>
#include "TMath.h"
#include "TString.h"
#include <string>
#include <map>



void vmprojections(){

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




  TCanvas* c1 = new TCanvas("c1","Track performance",200,10,700,800);
  c1->Divide(2,3);
  c1->SetFillColor(0);
  c1->SetGrid();

  TCanvas* c2 = new TCanvas("c2","Track performance",200,10,700,800);
  c2->Divide(2,3);
  c2->SetFillColor(0);
  c2->SetGrid();

  double max=40.0;
  
  TH1 *hist_L1 = new TH1F("L1","Projections per VM in L1",max+1,-0.5,max+0.5);
  TH1 *hist_L2 = new TH1F("L2","Projections per VM in L2",max+1,-0.5,max+0.5);
  TH1 *hist_L3 = new TH1F("L3","Projections per VM in L3",max+1,-0.5,max+0.5);
  TH1 *hist_L4 = new TH1F("L4","Projections per VM in L4",max+1,-0.5,max+0.5);
  TH1 *hist_L5 = new TH1F("L5","Projections per VM in L5",max+1,-0.5,max+0.5);
  TH1 *hist_L6 = new TH1F("L6","Projections per VM in L6",max+1,-0.5,max+0.5);


  TH1 *hist_D1 = new TH1F("D1","Projections per VM in D1",max+1,-0.5,max+0.5);
  TH1 *hist_D2 = new TH1F("D2","Projections per VM in D2",max+1,-0.5,max+0.5);
  TH1 *hist_D3 = new TH1F("D3","Projections per VM in D3",max+1,-0.5,max+0.5);
  TH1 *hist_D4 = new TH1F("D4","Projections per VM in D4",max+1,-0.5,max+0.5);
  TH1 *hist_D5 = new TH1F("D5","Projections per VM in D5",max+1,-0.5,max+0.5);


  ifstream in("vmprojections.txt");

  int count=0;

  std::map<TString, TH1*> hists;

  while (in.good()){

    TString name;
    int projs;
   
    in >>name>>projs;

    if (name.Contains("L1PHI")) hist_L1->Fill(projs);
    if (name.Contains("L2PHI")) hist_L2->Fill(projs);
    if (name.Contains("L3PHI")) hist_L3->Fill(projs);
    if (name.Contains("L4PHI")) hist_L4->Fill(projs);
    if (name.Contains("L5PHI")) hist_L5->Fill(projs);
    if (name.Contains("L6PHI")) hist_L6->Fill(projs);
    
    if (name.Contains("D1PHI")) hist_D1->Fill(projs);
    if (name.Contains("D2PHI")) hist_D2->Fill(projs);
    if (name.Contains("D3PHI")) hist_D3->Fill(projs);
    if (name.Contains("D4PHI")) hist_D4->Fill(projs);
    if (name.Contains("D5PHI")) hist_D5->Fill(projs);

    if (!in.good()) continue;
    //cout << name <<" "<< countall <<" "<< countpass << endl;

    if (projs>max) projs=max;

    std::map<TString, TH1*>::iterator it=hists.find(name);

    if (it==hists.end()) {
      TH1 *hist = new TH1F(name,name,max+1,-0.5,max+0.5);
      hist->Fill(projs);
      hists[name]=hist;
    } else {
      hists[name]->Fill(projs);
    }

    
    count++;
    
  }

  cout << "count = "<<count<<endl;
  
  c1->cd(1);
  gPad->SetLogy();
  //TF1* f1 = new TF1("f1","[0]*TMath::Power([1],x)*(TMath::Exp(-[1]))/TMath::Gamma(x+1)", 0, 30);
  //f1->SetParameters(1.0, 1.0); 
  //hist_L1L2_L3->Fit("f1", "R"); 
  hist_L1->Draw();

  c1->cd(2);
  gPad->SetLogy();
  hist_L2->Draw();

  c1->cd(3);
  gPad->SetLogy();
  hist_L3->Draw();

  c1->cd(4);
  gPad->SetLogy();
  hist_L4->Draw();

  c1->cd(5);
  gPad->SetLogy();
  hist_L5->Draw();

  c1->cd(6);
  gPad->SetLogy();
  hist_L6->Draw();

  c1->Print("vmprojections.pdf(", "pdf");

  c2->cd(1);
  gPad->SetLogy();
  hist_D1->Draw();
  
  c2->cd(2);
  gPad->SetLogy();
  hist_D2->Draw();

  c2->cd(3);
  gPad->SetLogy();
  hist_D3->Draw();

  c2->cd(4);
  gPad->SetLogy();
  hist_D4->Draw();

  c2->cd(5);
  gPad->SetLogy();
  hist_D5->Draw();

  c2->Print("vmprojections.pdf", "pdf");
  
  int pages=0;

  std::map<TString, TH1*>::iterator it=hists.begin();
  
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
    it->second->Draw();
    
    pages++;


    if (pages%4==0) {
      c->Print("vmprojections.pdf","pdf");
    }

    ++it;

    if (it==hists.end()&&(pages%4!=0)) {
      c->Print("vmprojections.pdf)","pdf");
    }

  }
  
  c->Print("vmprojections.pdf)","pdf");
  
}
