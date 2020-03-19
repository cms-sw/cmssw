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



void z0_and_rinv(){
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

 
 TH1 *hist1 = new TH1F("h1","z0 for accepted stub pairs L1L2",50,-50.0,50.0);
 TH1 *hist2 = new TH1F("h2","rinv for acceptes stub pairs L1L2",50,-0.01,0.01);
 TH1 *hist3 = new TH1F("h3","z0 for accepted stub pairs L3L4",50,-50.0,50.0);
 TH1 *hist4 = new TH1F("h4","rinv for acceptes stub pairs L3L4",50,-0.01,0.01);
 TH1 *hist5 = new TH1F("h5","z0 for accepted stub pairs L5L6",50,-50.0,50.0);
 TH1 *hist6 = new TH1F("h6","rinv for acceptes stub pairs L5L6",50,-0.01,0.01);


 ifstream in("z0_and_rinv.txt");

 int count=0;

 while (in.good()){

   double z0,rinv;
   int layer;
   
   in>>layer>>z0>>rinv;

   if (!in.good()) continue;

   if (layer==1) {
     hist1->Fill(z0);
     hist2->Fill(rinv);
   }

   if (layer==3) {
     hist3->Fill(z0);
     hist4->Fill(rinv);
   }

   if (layer==5) {
     hist5->Fill(z0);
     hist6->Fill(rinv);
   }

   count++;

 }

 cout << "Processed: "<<count<<" events"<<endl;

 c1->cd(1);
 hist1->Draw();

 c1->cd(2);
 hist2->Draw();

 c1->cd(3);
 hist3->Draw();

 c1->cd(4);
 hist4->Draw();

 c1->cd(5);
 hist5->Draw();

 c1->cd(6);
 hist6->Draw();

 c1->Print("z0_and_rinv.png");
 c1->Print("z0_and_rinv.pdf");


}

