{
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




 c1 = new TCanvas("c1","Track performance",200,10,700,800);
 c1->Divide(1,1);
 c1->SetFillColor(0);
 c1->SetGrid();

 
 TH1 *hist1 = new TH1F("h1","pt stub old",50,0.0,50.0);


 ifstream in("stubptL1_old.txt");

 int count=0;

 while (in.good()){

   double  pt;

   in >>pt;

   hist1->Fill(fabs(pt));
   hist1->Fill(fabs(pt));
   hist1->Fill(fabs(pt));
   hist1->Fill(fabs(pt));
   hist1->Fill(fabs(pt));

   count++;

 }

 TH1 *hist2 = new TH1F("h2","pt stub new",50,0.0,50.0);


 ifstream innew("stubptL1_new.txt");

 count=0;

 while (innew.good()){

   double  pt;

   innew >>pt;

   hist2->Fill(fabs(pt));
   count++;

 }


 cout << "Processed: "<<count<<" events"<<endl;

 c1->cd(1);
 h2->SetLineColor(kRed);
 h2->Draw();
 h1->SetLineColor(kBlue);
 h1->Draw("SAME");

 c1->Print("stubptL1.pdf");

}

