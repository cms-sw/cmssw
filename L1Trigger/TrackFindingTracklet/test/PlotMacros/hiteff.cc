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



void hiteff(){
//
// To see the output of this macro, click here.

//

#include "TMath.h"

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

 
 TH1 *hist1 = new TH1F("eff","eff",6,0.5,6.5);

 TH1 *hist11 = new TH1F("Layers hit","Layers hit",7,-0.5,6.5);
 
 TH1 *hist21 = new TH1F("Layer 1 Eff","Layer 1 Eff",50,-2.5,2.5);
 TH1 *hist31 = new TH1F("Layer 1 Eff num.","Layer 1 Eff num",50,-2.5,2.5);

 TH1 *hist22 = new TH1F("Layer 2 Eff","Layer 2 Eff",50,-2.5,2.5);
 TH1 *hist32 = new TH1F("Layer 2 Eff num.","Layer 2 Eff num",50,-2.5,2.5);

 TH1 *hist23 = new TH1F("Layer 3 Eff","Layer 3 Eff",50,-2.5,2.5);
 TH1 *hist33 = new TH1F("Layer 3 Eff num.","Layer 3 Eff num",50,-2.5,2.5);

 TH1 *hist24 = new TH1F("Layer 4 Eff","Layer 4 Eff",50,-2.5,2.5);
 TH1 *hist34 = new TH1F("Layer 4 Eff num.","Layer 4 Eff num",50,-2.5,2.5);

 TH1 *hist25 = new TH1F("Layer 5 Eff","Layer 5 Eff",50,-2.5,2.5);
 TH1 *hist35 = new TH1F("Layer 5 Eff num.","Layer 5 Eff num",50,-2.5,2.5);

 TH1 *hist26 = new TH1F("Layer 6 Eff","Layer 6 Eff",50,-2.5,2.5);
 TH1 *hist36 = new TH1F("Layer 6 Eff num.","Layer 6 Eff num",50,-2.5,2.5);


 ifstream in("hiteff.txt");

 int count=0;

 int l1,l2,l3,l4,l5,l6;
 double eta;

 while (in.good()){

   in >>eta>>l1>>l2>>l3>>l4>>l5>>l6;
   
   if (!in.good()) continue;

   int nhits=0;

   if (l1>0) {
     hist1->Fill(1.0);
     nhits++;
   }
   if (l2>0) {
     hist1->Fill(2.0);
     nhits++;
   }
   if (l3>0) {
     hist1->Fill(3.0);
     nhits++;
   }
   if (l4>0) {
     hist1->Fill(4.0);
     nhits++;
   }
   if (l5>0) {
     hist1->Fill(5.0);
     nhits++;
   }
   if (l6>0) {
     hist1->Fill(6.0);
     nhits++;
   }

   hist11->Fill(nhits);

   hist31->Fill(eta);
   if (l1>0) hist21->Fill(eta);

   hist32->Fill(eta);
   if (l2>0) hist22->Fill(eta);

   hist33->Fill(eta);
   if (l3>0) hist23->Fill(eta);

   hist34->Fill(eta);
   if (l4>0) hist24->Fill(eta);

   hist35->Fill(eta);
   if (l5>0) hist25->Fill(eta);

   hist36->Fill(eta);
   if (l6>0) hist26->Fill(eta);
   
   count++;

 }

 cout << "Processed: "<<count<<" events"<<endl;

 //c1->cd(1);
 //hist1->Scale(1.0/count);
 //hist1->SetMinimum(0.0);
 //hist1->Draw();

 //c1->cd(2);
 //hist11->Draw();


 c1->cd(1);
 hist21->Divide(hist31);
 hist21->SetMinimum(0.0);
 hist21->SetMaximum(1.05);
 hist21->Draw();

 c1->cd(2);
 hist22->Divide(hist32);
 hist22->SetMinimum(0.0);
 hist22->SetMaximum(1.05);
 hist22->Draw();
 
 c1->cd(3);
 hist23->Divide(hist33);
 hist23->SetMinimum(0.0);
 hist23->SetMaximum(1.05);
 hist23->Draw();
 
 c1->cd(4);
 hist24->Divide(hist34);
 hist24->SetMinimum(0.0);
 hist24->SetMaximum(1.05);
 hist24->Draw();
 
 c1->cd(5);
 hist25->Divide(hist35);
 hist25->SetMinimum(0.0);
 hist25->SetMaximum(1.05);
 hist25->Draw();
 
 c1->cd(6);
 hist26->Divide(hist36);
 hist26->SetMinimum(0.0);
 hist26->SetMaximum(1.05);
 hist26->Draw();
 
 c1->Print("hiteff.pdf");

}

