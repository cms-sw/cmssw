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


void fittrack(){
//
// To see the output of this macro, click here.

//

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


 TCanvas* c1 = new TCanvas("c1","Track performance",200,10,700,400);
 c1->Divide(2,1);
 c1->SetFillColor(0);
 c1->SetGrid();


 double max=128.0;
 
 TH1 *hist1 = new TH1F("h1","Tracklets with atleast one match",max+1,-0.5,max+0.5);
 TH1 *hist2 = new TH1F("h1","Tracklets with two or more matches",max+1,-0.5,max+0.5);


 ifstream in("fittrack.txt");

 int count=0;

 std::map<TString, TH1*> hists1;
 std::map<TString, TH1*> hists2;

 while (in.good()){

   TString name;
   int countall,countfit;
   
   in >>name>>countall>>countfit;

   if (!in.good()) continue;

   if (countall>max) countall=max;
   if (countfit>max) countfit=max;

   hist1->Fill(countfit);
   hist2->Fill(countall);

   std::map<TString, TH1*>::iterator it=hists1.find(name);

   if (it==hists1.end()) {
     TH1 *hist = new TH1F(name,name,max+1,-0.5,max+0.5);
     hist->Fill(countfit);
     hists1[name]=hist;
   } else {
     hists1[name]->Fill(countfit);
   }


   std::map<TString, TH1*>::iterator it2=hists2.find(name);

   if (it2==hists2.end()) {
     TH1 *hist = new TH1F(name,name,max+1,-0.5,max+0.5);
     hist->Fill(countall);
     hists2[name]=hist;
   } else {
     hists2[name]->Fill(countall);
   }


   count++;

 }

 cout << "count = "<<count<<endl;

 c1->cd(1);
 gPad->SetLogy(); 
 hist1->SetLineColor(kBlue);
 hist1->Draw();

 hist2->SetLineColor(kRed);
 hist2->Draw("Same");

 c1->cd(2);
 hist1->SetLineColor(kBlue);
 hist1->Draw();

 hist2->SetLineColor(kRed);
 hist2->Draw("Same");

 c1->Print("fittracksummary.pdf");


 int pages=0;

 std::map<TString, TH1*>::iterator it=hists1.begin();

 TCanvas* c=0;

 bool first=true;

 while(it!=hists1.end()) {

   if (pages%4==0) {
     
     c = new TCanvas(it->first,"Track performance",200,50,600,700);
     c->Divide(2,2);
     c->SetFillColor(0);
     c->SetGrid();

   }

   c->cd(pages%4+1);
   gPad->SetLogy();
   it->second->SetLineColor(kBlue);
   plotHist(it->second,0.05,ncut1,ncut2);
   
   hists2[it->first]->SetLineColor(kRed);
   hists2[it->first]->Draw("Same");


   pages++;

   if (pages%4==0) {
     if (first) {
       first=false;
       c->Print("fittrack.pdf(","pdf");
     }
     else{
       c->Print("fittrack.pdf","pdf");
     }
   }

   ++it;

   if (it==hists1.end()&&(pages%4!=0)) {
     c->Print("fittrack.pdf)","pdf");
   }

 }


}
