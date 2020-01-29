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
#include "TText.h"
#include <string>
#include <map>


#include "plotHist.cc"


  

void dtccounts(){
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



 TCanvas* c1 = new TCanvas("c1","Track performance",200,10,500,800);
 c1->Divide(1,1);
 c1->SetFillColor(0);
 c1->SetGrid();


 
 double max=300.0;

 TH1F* hist01=new TH1F("All DTCs","All DTCs",max+1,-0.5,max+0.5);

 
 ifstream in("dtccounts.txt");

 int count=0;

 std::map<string, TH1*> hists;

 while (in.good()){

   string dtc;
   int stubs;
  
   in >>dtc>>stubs;

   if (!in.good()) continue;

   if (stubs>max) stubs=max;

   std::map<string, TH1*>::iterator it=hists.find(dtc);

   if (it==hists.end()) {
     TString name="DTC = ";
     name+=dtc;
     TH1 *hist = new TH1F(name,name,max+1,-0.5,max+0.5);
     hists[dtc]=hist;
   } 
   hists[dtc]->Fill(stubs);
   
   hist01->Fill(stubs);

   count++;

 }

 cout << "count = "<<count<<endl;

 c1->cd(1);
 hist01->Draw();
 c1->Print("dtccounts.pdf(","pdf");


 int pages=0;

 std::map<string, TH1*>::iterator it=hists.begin();

 TCanvas* c=0;

 bool closed=false;
 
 while(it!=hists.end()) {

   if (pages%4==0) {

     TString name = "DTC link : ";
     name+=it->first;
     
     c = new TCanvas(name,"Track performance",200,50,600,700);
     c->Divide(2,2);
     c->SetFillColor(0);
     c->SetGrid();

   }

   c->cd(pages%4+1);
   //sgPad->SetLogy();
   //plotHist(it->second,0.03,83.0,50.0);
   it->second->Draw();
   
   pages++;

   ++it;

   if (pages%4==0) {
     if (it!=hists.end()) {
       c->Print("dtccounts.pdf","pdf");
     }
     else {
       closed=true;
       c->Print("dtccounts.pdf)","pdf");
     }
   }

 }

 if (!closed) {
   c->Print("dtccounts.pdf)","pdf");
 }
   
}
