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



void nmatches(){
//
// To see the output of this macro, click here.

//

#include "TMath.h"

gROOT->Reset();

gROOT->SetStyle("Plain");

gStyle->SetCanvasColor(kWhite);

gStyle->SetCanvasBorderMode(0);     // turn off canvas borders
gStyle->SetPadBorderMode(0);
gStyle->SetOptStat(0);
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

 TH1 *hist1 = new TH1F("h1","nMatches L1L2",7,-0.5,6.5);
 TH1 *hist2 = new TH1F("h2","nMatches L3L4",7,-0.5,6.5);
 TH1 *hist3 = new TH1F("h3","nMatches L5L6",7,-0.5,6.5);
 TH1 *hist4 = new TH1F("h4","nMatches L1D1",7,-0.5,6.5);

 TH1 *hist5 = new TH1F("h5","nMatches D1D2",7,-0.5,6.5);
 TH1 *hist6 = new TH1F("h6","nMatches D3D4",7,-0.5,6.5);

 
 int max=50;
 TH1 *hist11 = new TH1F("h11","nMatches L1L2",max+1,-0.5,max+0.5);
 TH1 *hist12 = new TH1F("h12","nMatches L3L4",max+1,-0.5,max+0.5);
 TH1 *hist13 = new TH1F("h13","nMatches L5L6",max+1,-0.5,max+0.5);



 ifstream in("nmatches.txt");

 int count=0;

 while (in.good()){

   int layer,disk,nmatchlayer,nmatchdisk;
   
   in>>layer>>disk>>nmatchlayer>>nmatchdisk;

   int nmatch=nmatchlayer+nmatchdisk;

   if (!in.good()) continue;

   if (disk==0) {
     if (layer==1) {
       hist1->Fill(nmatch);
     }
     
     if (layer==3) {
       hist2->Fill(nmatch);
     }

     if (layer==5) {
       hist3->Fill(nmatch);
     }
   }

   if ((layer!=0) && (disk!=0)) {
     hist4->Fill(nmatch);
   }
   
   if (layer==0) {
     if (abs(disk)==1) {
       hist5->Fill(nmatch);
     }
     
     if (abs(disk)==3) {
       hist6->Fill(nmatch);
     }
   }
   
   count++;

 }


 ifstream in2("nmatchessector.txt");

 int count2=0;

 while (in2.good()){

   int matchesL1,matchesL3,matchesL5;
   
   in2>>matchesL1>>matchesL3>>matchesL5;

   if (!in2.good()) continue;

   if (matchesL1>max) matchesL1=max;
   if (matchesL3>max) matchesL3=max;
   if (matchesL5>max) matchesL5=max;

   hist11->Fill(matchesL1);
   hist12->Fill(matchesL3);
   hist13->Fill(matchesL5);

   count2++;

 }

 double scale=1.0/(100*27);
 scale=1.0;
 
 cout << "Processed: "<<count2<<" events"<<endl;

 c1->cd(1);
 hist1->Scale(scale);
 hist1->Draw();

 c1->cd(2);
 hist2->Scale(scale);
 hist2->Draw();

 c1->cd(3);
 hist3->Scale(scale);
 hist3->Draw();

 c1->cd(4);
 hist4->Scale(scale);
 hist4->Draw();

 c1->cd(5);
 hist5->Scale(scale);
 hist5->Draw();

 c1->cd(6);
 hist6->Scale(scale);
 hist6->Draw();

 c1->Print("nmatches.png");
 c1->Print("nmatches.pdf");

 c2->cd(1);
 hist11->Draw();

 c2->cd(2);
 hist12->Draw();

 c2->cd(3);
 hist13->Draw();

 c2->Print("nmatchessector.png");
 c2->Print("nmatchessector.pdf");



}

