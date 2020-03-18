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



void trackprojocc(){
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




 TCanvas* c1 = new TCanvas("c1","Track performance",200,10,700,800);
 c1->Divide(2,2);
 c1->SetFillColor(0);
 c1->SetGrid();

 double max=100.0;
 
 TH1 *hist1 = new TH1F("h1","TrackProjOcc",max+1,-0.5,max+0.5);

 ofstream out("trackprojocc.dat");
 
 ifstream in("trackprojocc.txt");

 int count=0;

 std::map<TString, TH1* > hists;

 while (in.good()){

   TString name;
   int trackprojocc;
  
   in >>name>>trackprojocc;

   if (!in.good()) continue;
   //cout << name <<" "<< countall <<" "<< countpass << endl;

   if (trackprojocc>max) trackprojocc=max;

   hist1->Fill(trackprojocc);


   std::map<TString, TH1* >::iterator it=hists.find(name);

   if (it==hists.end()) {
     TString name1=name;
     TH1 *hist = new TH1F(name1,name1,max+1,-0.5,max+0.5);
     hist->Fill(trackprojocc);
     hists[name]=hist;
   } else {
     hists[name]->Fill(trackprojocc);
   }

   count++;

 }

 cout << "count = "<<count<<endl;

 c1->cd(1);
 hist1->SetLineColor(kBlue);
 hist1->Draw();

 c1->cd(2);
 gPad->SetLogy();
 hist1->Draw();

 c1->Print("trackprojoccsum.pdf");

 int pages=0;

 std::map<TString, TH1* >::iterator it=hists.begin();

 TCanvas* c=0;

 bool first=true;

 while(it!=hists.end()) {

   if (pages%4==0) {
     
     c = new TCanvas(it->first,"Track performance",200,50,600,700);
     c->Divide(2,2);
     c->SetFillColor(0);
     c->SetGrid();

   }

   c->cd(pages%4+1);
   gPad->SetLogy();
   it->second->SetLineColor(kRed);
   if (it->second->GetMean()==0.0) {
     out << "Is empty : "<<it->first << endl;
   } else {
     out << "Not empty : "<<it->first << " "<<it->second->GetMean()<<endl;
   }
   
   it->second->Draw();

   pages++;

   if (pages%4==0) {
     if (first) {
       first=false;
       c->Print("trackprojocc.pdf(","pdf");
     }
     else{
       c->Print("trackprojocc.pdf","pdf");
     }
   }

   ++it;


 }

 c->Print("trackprojocc.pdf)","pdf");


}
