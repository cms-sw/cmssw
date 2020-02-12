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
#include <string>
#include <map>



void projectiontransceiver(){

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
 c1->Divide(2,2);
 c1->SetFillColor(0);
 c1->SetGrid();

 double max=200.0;
 
 TH1 *hist1 = new TH1F("h1","Projections Minus",100,0.0,max);
 TH1 *hist2 = new TH1F("h2","Projections Plus",100,0.0,max);


 ifstream in("projectiontransceiver.txt");

 int count=0;

 while (in.good()){

   int nproj;
   TString name;

   in >>name>>nproj;

   if (nproj>max) nproj=max-1;

   //cout <<layer<<" "<<r<<endl;

   if (name.Contains("Minus")) hist1->Fill(nproj);
   if (name.Contains("Plus")) hist2->Fill(nproj);
 
   count++;

 }

 cout << "Processed: "<<count<<" events"<<endl;

 c1->cd(1);
 hist1->Draw();

 c1->cd(2);
 gPad->SetLogy();
 hist1->Draw();

 c1->cd(3);
 hist2->Draw();

 c1->cd(4);
 gPad->SetLogy();
 hist2->Draw();

 c1->Print("projectiontransceiver.png");
 c1->Print("projectiontransceiver.pdf");

}

