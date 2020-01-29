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



void inputlink(){
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




 TCanvas* c1 = new TCanvas("c1","Track performance",200,10,700,400);
 c1->Divide(2,1);
 c1->SetFillColor(0);
 c1->SetGrid();


 double max=128.0;
 
 TH1 *hist1 = new TH1F("h1","Stubs per input link memory",max+1,-0.5,max+0.5);


 ifstream in("inputlink.txt");

 int count=0;

 std::map<TString, TH1*> hists1;

 while (in.good()){

   TString name;
   int nstubs;
   
   in >>name>>nstubs;

   if (!in.good()) continue;

   if (nstubs>max) nstubs=max;

   hist1->Fill(nstubs);

   std::map<TString, TH1*>::iterator it=hists1.find(name);

   if (it==hists1.end()) {
     TH1 *hist = new TH1F(name,name,max+1,-0.5,max+0.5);
     hist->Fill(nstubs);
     hists1[name]=hist;
   } else {
     hists1[name]->Fill(nstubs);
   }

   count++;

 }

 cout << "count = "<<count<<endl;

 c1->cd(1);
 gPad->SetLogy(); 
 hist1->SetLineColor(kBlue);
 hist1->Draw();


 c1->cd(2);
 hist1->SetLineColor(kBlue);
 hist1->Draw();

 c1->Print("inputlinksummary.pdf");


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
   it->second->Draw();
   

   pages++;

   if (pages%4==0) {
     if (first) {
       first=false;
       c->Print("inputlink.pdf(","pdf");
     }
     else{
       c->Print("inputlink.pdf","pdf");
     }
   }

   ++it;

   if (it==hists1.end()&&(pages%4!=0)) {
     c->Print("inputlink.pdf)","pdf");
   }

 }


}
