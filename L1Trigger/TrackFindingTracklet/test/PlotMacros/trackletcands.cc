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


void trackletcands(){
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
 c1->Divide(2,2);
 c1->SetFillColor(0);
 c1->SetGrid();

 
 TH1 *hist1 = new TH1F("h1","Number of tracklet candidates L1+L2 per VM pair",100,0.0,100.0);

 TH1 *hist2 = new TH1F("h2","Number of tracklet candidates L3+L4 per VM pair",100,0.0,100.0);

 TH1 *hist3 = new TH1F("h3","Number of tracklet candidates L5+L6 per VM pair",100,0.0,100.0);


 ifstream in("trackletcands.txt");

 int count=0;

 while (in.good()){

   int layer,npairs;
  

   in >>layer>>npairs;

   //cout <<layer<<" "<<r<<endl;

   if (layer==1) hist1->Fill(npairs);
   if (layer==3) hist2->Fill(npairs);
   if (layer==5) hist3->Fill(npairs);

   count++;

 }

 cout << "Processed: "<<count<<" events"<<endl;

 c1->cd(1);
 plotfcn(hist1);

 c1->cd(2);
 plotfcn(hist2);

 c1->cd(3);
 plotfcn(hist3);



 c1->Print("trackletcands.png");
 c1->Print("trackletcands.pdf");

}

