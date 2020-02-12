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
 c1->Divide(2,2);
 c1->SetFillColor(0);
 c1->SetGrid();

 
 TH1 *hist1 = new TH1F("h1","Number of stub pairs in D1+D2 per VM",100,0.0,100.0);
 TH1 *hist2 = new TH1F("h2","Number of stub pairs in D3+D4 per VM",100,0.0,100.0);


 ifstream in("stubpairsdisk.txt");

 int count=0;

 int disk,npairs;
 
 in >>disk>>npairs;

 while (in.good()){

   if (disk==1) hist1->Fill(npairs);
   if (disk==3) hist2->Fill(npairs);

   in >>disk>>npairs;

   count++;

 }

 cout << "Processed: "<<count<<" events"<<endl;

 c1->cd(1);
 gPad->SetLogy();
 h1->Draw();

 c1->cd(2);
 gPad->SetLogy();
 h2->Draw();


 c1->Print("stubpairsdisk.png");
 c1->Print("stubpairsdisk.pdf");

}

