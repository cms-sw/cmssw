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

#include "plotfcn.h"



void vmmatches(){
//
// To see the output of this macro, click here.

//


gROOT->Reset();

gROOT->SetStyle("Plain");

gStyle->SetCanvasColor(kWhite);

gStyle->SetCanvasBorderMode(0);     // turn off canvas borders
gStyle->SetPadBorderMode(0);
gStyle->SetOptStat(0);
gStyle->SetOptTitle(1);

  // For publishing:
  gStyle->SetLineWidth(1.5);
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

 int max=150;
 
 TH1 *hist1 = new TH1F("h1","Matches per VM L1",max+1,-0.5,max+0.5);
 TH1 *hist2 = new TH1F("h2","Matches per VM L2",max+1,-0.5,max+0.5);
 TH1 *hist3 = new TH1F("h3","Matches per VM L3",max+1,-0.5,max+0.5);
 TH1 *hist4 = new TH1F("h4","Matches per VM L4",max+1,-0.5,max+0.5);
 TH1 *hist5 = new TH1F("h5","Matches per VM L5",max+1,-0.5,max+0.5);
 TH1 *hist6 = new TH1F("h6","Matches per VM L6",max+1,-0.5,max+0.5);


 ifstream in("vmmatches.txt");

 int count=0;

 while (in.good()){

   int layer,nmatch,i,j;
  
   in >>layer>>i>>j>>nmatch;

   //if (nmatch==0) continue;

   if (nmatch>max) nmatch=max;

   //cout <<layer<<" "<<r<<endl;

   if (layer==1) hist1->Fill(nmatch);
   if (layer==2) hist2->Fill(nmatch);
   if (layer==3) hist3->Fill(nmatch);
   if (layer==4) hist4->Fill(nmatch);
   if (layer==5) hist5->Fill(nmatch);
   if (layer==6) hist6->Fill(nmatch);

   count++;

 }

 cout << "Processed: "<<count<<" events"<<endl;

 c1->cd(1);
 plotfcn(hist1);

 c1->cd(2);
 plotfcn(hist2);

 c1->cd(3);
 plotfcn(hist3);

 c1->cd(4);
 plotfcn(hist4);

 c1->cd(5);
 plotfcn(hist5);

 c1->cd(6);
 plotfcn(hist6);

 c1->Print("vmmatches.png");
 c1->Print("vmmatches.pdf");

}

