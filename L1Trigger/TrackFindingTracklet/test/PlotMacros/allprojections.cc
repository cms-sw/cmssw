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



void allprojections(){

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
  c1->Divide(2,3);
  c1->SetFillColor(0);
  c1->SetGrid();
 
  TCanvas* c2 = new TCanvas("c2","Track performance",200,10,700,800);
  c2->Divide(2,3);
  c2->SetFillColor(0);
  c2->SetGrid();

  int ncut1=72;
  int ncut2=108;

  double max=128.0;
 
  TH1 *hist1 = new TH1F("h1","Projections L1",max,+0.5,max+0.5);
  TH1 *hist2 = new TH1F("h2","Projections L2",max,+0.5,max+0.5);
  TH1 *hist3 = new TH1F("h3","Projections L3",max,+0.5,max+0.5);
  TH1 *hist4 = new TH1F("h4","Projections L4",max,+0.5,max+0.5);
  TH1 *hist5 = new TH1F("h5","Projections L5",max,+0.5,max+0.5);
  TH1 *hist6 = new TH1F("h6","Projections L6",max,+0.5,max+0.5);

  TH1 *hist10 = new TH1F("h10","Projections D1",max,+0.5,max+0.5);
  TH1 *hist11 = new TH1F("h11","Projections D2",max,+0.5,max+0.5);
  TH1 *hist12 = new TH1F("h12","Projections D3",max,+0.5,max+0.5);
  TH1 *hist13 = new TH1F("h13","Projections D4",max,+0.5,max+0.5);
  TH1 *hist14 = new TH1F("h14","Projections D5",max,+0.5,max+0.5);

  ifstream in("allprojections.txt");

  int count=0;

  std::map<TString, TH1*> hists;

  while (in.good()){

    TString name;
    int projs;
   
    in >>name>>projs;

    if (!in.good()) continue;
    //cout << name <<" "<< countall <<" "<< countpass << endl;
    
    if (projs>max) projs=max;

    if (name.Contains("L1PHI")) hist1->Fill(projs);
    if (name.Contains("L2PHI")) hist2->Fill(projs);
    if (name.Contains("L3PHI")) hist3->Fill(projs);
    if (name.Contains("L4PHI")) hist4->Fill(projs);
    if (name.Contains("L5PHI")) hist5->Fill(projs);
    if (name.Contains("L6PHI")) hist6->Fill(projs);

    if (name.Contains("D1PHI")) hist10->Fill(projs);
    if (name.Contains("D2PHI")) hist11->Fill(projs);
    if (name.Contains("D3PHI")) hist12->Fill(projs);
    if (name.Contains("D4PHI")) hist13->Fill(projs);
    if (name.Contains("D5PHI")) hist14->Fill(projs);



   std::map<TString, TH1*>::iterator it=hists.find(name);

   if (it==hists.end()) {
     TH1 *hist = new TH1F(name,name,max,+0.5,max+0.5);
     hist->Fill(projs);
     hists[name]=hist;
   } else {
     hists[name]->Fill(projs);
   }


   count++;

 }

 cout << "count = "<<count<<endl;

 c1->cd(1);
 //gPad->SetLogy();
 plotHist(hist1,0.05,ncut1,ncut2);

 c1->cd(2);
 //gPad->SetLogy();
 plotHist(hist2,0.05,ncut1,ncut2);


 c1->cd(3);
 //gPad->SetLogy();
 plotHist(hist3,0.05,ncut1,ncut2);


 c1->cd(4);
 //gPad->SetLogy();
 plotHist(hist4,0.05,ncut1,ncut2);


 c1->cd(5);
 //gPad->SetLogy();
 plotHist(hist5,0.05,ncut1,ncut2);


 c1->cd(6);
 //gPad->SetLogy();
 plotHist(hist6,0.05,ncut1,ncut2);


 c1->Print("allprojections.pdf(","pdf");

 c2->cd(1);
 //gPad->SetLogy();
 plotHist(hist10,0.05,ncut1,ncut2);


 c2->cd(2);
 //gPad->SetLogy();
 plotHist(hist11,0.05,ncut1,ncut2);

 c2->cd(3);
 //gPad->SetLogy();
 plotHist(hist12,0.05,ncut1,ncut2);

 c2->cd(4);
 //gPad->SetLogy();
 plotHist(hist13,0.05,ncut1,ncut2);

 c2->cd(5);
 //gPad->SetLogy();
 plotHist(hist14,0.05,ncut1,ncut2);


 c2->Print("allprojections.pdf","pdf");


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
   //gPad->SetLogy();
   plotHist(it->second,0.03,ncut1,ncut2);

   pages++;

   if (pages%4==0) {
     c->Print("allprojections.pdf","pdf");
   }

   ++it;

   if (it==hists.end()&&(pages%4!=0)) {
     c->Print("allprojections.pdf)","pdf");
   }

 }

 c->Print("allprojections.pdf)","pdf");
 
}
