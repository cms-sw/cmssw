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



void trackpars(){
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

 
 TH1 *hist10 = new TH1F("h10","eta for all tracks",50,-2.5,2.5);
 TH1 *hist11 = new TH1F("h11","phi0 for all tracks",50,-3.1415,3.1415);
 TH1 *hist12 = new TH1F("h12","z0 for all tracks",50,-25.0,25.0);
 TH1 *hist13 = new TH1F("h13","phi in sector all tracks",50,0.0,1.0);
 TH1 *hist14 = new TH1F("h14","rinv in sector all tracks",50,-0.006,0.006);


 TH1 *hist20 = new TH1F("h20","eta for non-dupliates",50,-2.5,2.5);
 TH1 *hist21 = new TH1F("h21","phi0 for non-duplicates",50,-3.1415,3.1415);
 TH1 *hist22 = new TH1F("h22","z0 for non-duplidates",50,-25.0,25.0);
 TH1 *hist23 = new TH1F("h23","phi in sector non-duplicates",50,0.0,1.0);
 TH1 *hist24 = new TH1F("h24","rinv for non-duplicates",50,-0.006,0.006);


 ifstream in("trackpars.txt");

 int count=0;

 while (in.good()) {

   double eta,phi0,z0,phisect,rinv;
   int duplicate;
   
   in>>duplicate>>eta>>phi0>>z0>>phisect>>rinv;

   if (!in.good()) continue;

   hist10->Fill(eta);
   hist11->Fill(phi0);
   hist12->Fill(z0);
   hist13->Fill(phisect);
   hist14->Fill(rinv);
   if (!duplicate) {
     hist20->Fill(eta);
     hist21->Fill(phi0);
     hist22->Fill(z0);
     hist23->Fill(phisect);
     hist24->Fill(rinv);
   }

   count++;

 }

//cout << "Processed: "<<count<<" events"<<endl;

 c1->cd(1);
 hist10->SetMinimum(0);
 hist10->SetLineColor(kBlue);
 hist10->Draw();
 hist20->SetLineColor(kRed);
 hist20->Draw("Same");

 c1->cd(2);
 hist11->SetMinimum(0);
 hist11->SetLineColor(kBlue);
 hist11->Draw();
 hist21->SetLineColor(kRed);
 hist21->Draw("Same");

 c1->cd(3);
 hist12->SetMinimum(0);
 hist12->SetLineColor(kBlue);
 hist12->Draw();
 hist22->SetLineColor(kRed);
 hist22->Draw("Same");

 c1->cd(4);
 hist13->SetMinimum(0);
 hist13->SetLineColor(kBlue);
 hist13->Draw();
 hist23->SetLineColor(kRed);
 hist23->Draw("Same");

 c1->cd(5);
 hist14->SetMinimum(0);
 hist14->SetLineColor(kBlue);
 hist14->Draw();
 hist24->SetLineColor(kRed);
 hist24->Draw("Same");
 

 
 c1->Print("trackpars.pdf");


}

