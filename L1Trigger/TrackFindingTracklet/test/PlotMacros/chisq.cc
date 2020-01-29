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



void chisq(){
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

 
 TH1 *hist1 = new TH1F("ln(1+chisq)","ln(1+chisq)",150,0.0,15.0);
 TH1 *hist2 = new TH1F("ln(1+ichisq)","ln(1+ichisq)",150,0.0,15.0);
 TH2 *hist3 = new TH2F("ln(1+chisq) vs ln(1+ichisq)","ln(1+ichisq) vs ln(1+ichisq)",60,0.0,15.0,60,0.0,15.0);
 TH2 *hist4 = new TH2F("ln(1+chisq)-ln(1+ichisq) vs eta","ln(1+ichisq) - ln(1+ichisq) vs eta",25,-2.5,2.5,23,-4.0,4.0);

 TH2 *hist13 = new TH2F("chisq vs ichisq","chisq vs ichisq",60,0.0,500.0,60,0.0,500.0);
 

 ifstream in("chisq.txt");

 int count=0;

 double eta,chisq,ichisq;

 while (in.good()){

   in >>eta>>chisq>>ichisq;
   
   if (!in.good()) continue;

   hist1->Fill(log(1.0+chisq));
   hist2->Fill(log(1.0+ichisq));

   hist3->Fill(log(1.0+chisq),log(1.0+ichisq));
   hist4->Fill(eta,log(1.0+chisq)-log(1.0+ichisq));

   hist13->Fill(chisq,ichisq);

   
   count++;

 }

 cout << "Processed: "<<count<<" events"<<endl;


 c1->cd(1);
 hist1->Draw();

 c1->cd(2);
 hist2->Draw();

 c1->cd(3);
 hist3->Draw("box");

 c1->cd(4);
 hist4->Draw("box");
 
 c1->Print("chisq.pdf");

}

