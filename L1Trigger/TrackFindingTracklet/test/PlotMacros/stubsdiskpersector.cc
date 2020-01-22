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



void stubsdiskpersector(){
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





 TCanvas* c4 = new TCanvas("c4","Track performance",200,10,700,1000);
 c4->Divide(2,3);
 c4->SetFillColor(0);
 c4->SetGrid();

 double max=200.0;
 
 TH1 *hist1 = new TH1F("h1","Number of stubs D1",50,0.0,max);
 TH1 *hist2 = new TH1F("h2","Number of stubs D2",50,0.0,max);
 TH1 *hist3 = new TH1F("h3","Number of stubs D3",50,0.0,max);
 TH1 *hist4 = new TH1F("h4","Number of stubs D4",50,0.0,max);
 TH1 *hist5 = new TH1F("h5","Number of stubs D5",50,0.0,max);


 ifstream in("stubsdiskpersector.txt");

 int count=0;

 int d1,d2,d3,d4,d5;
  
 in >>d1>>d2>>d3>>d4>>d5;

 while (in.good()){

   hist1->Fill(d1);
   hist2->Fill(d2);
   hist3->Fill(d3);
   hist4->Fill(d4);
   hist5->Fill(d5);

   in >>d1>>d2>>d3>>d4>>d5;
 
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

 c4->Print("stubsdiskpersector.png");
 c4->Print("stubsdiskpersector.pdf");


}

