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



void stubslayer(){
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





 TCanvas* c4 = new TCanvas("c4","Track performance",200,10,700,1000);
 c4->Divide(2,3);
 c4->SetFillColor(0);
 c4->SetGrid();

 double max=4000.0;
 
 TH1 *hist1 = new TH1F("h1","Number of stubs L1",50,0.0,max);
 TH1 *hist2 = new TH1F("h2","Number of stubs L2",50,0.0,max);
 TH1 *hist3 = new TH1F("h3","Number of stubs L3",50,0.0,max);
 TH1 *hist4 = new TH1F("h4","Number of stubs L4",50,0.0,max);
 TH1 *hist5 = new TH1F("h5","Number of stubs L5",50,0.0,max);
 TH1 *hist6 = new TH1F("h6","Number of stubs L6",50,0.0,max);


 ifstream in("stubslayer.txt");

 int count=0;

 int l1,l2,l3,l4,l5,l6;
  
 in >>l1>>l2>>l3>>l4>>l5>>l6;

 while (in.good()){

   hist1->Fill(l1);
   hist2->Fill(l2);
   hist3->Fill(l3);
   hist4->Fill(l4);
   hist5->Fill(l5);
   hist6->Fill(l6);

   in >>l1>>l2>>l3>>l4>>l5>>l6;
 
   count++;

 }

 cout << "Processed: "<<count<<" events"<<endl;

 c4->cd(1);
 hist1->Draw();

 c4->cd(2);
 hist2->Draw();

 c4->cd(3);
 hist3->Draw();

 c4->cd(4);
 hist4->Draw();

 c4->cd(5);
 hist5->Draw();

 c4->cd(6);
 hist6->Draw();

 c4->Print("stubslayer.png");
 c4->Print("stubslayer.pdf");


}

