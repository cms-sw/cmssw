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
  gStyle->SetTitleOffset(1.3,"y");
  gStyle->SetPadTopMargin(0.1);
  gStyle->SetPadRightMargin(0.1);
  gStyle->SetPadBottomMargin(0.16);
  gStyle->SetPadLeftMargin(0.15);




 c1 = new TCanvas("c1","Track performance",200,10,700,800);
 c1->Divide(2,3);
 c1->SetFillColor(0);
 c1->SetGrid();

 int maxtracklet=40;
 
 TH1 *hist1 = new TH1F("h1","L1+L2",maxtracklet+1,-0.5,maxtracklet+0.5);
 TH1 *hist2 = new TH1F("h2","L3+L4",maxtracklet+1,-0.5,maxtracklet+0.5);
 TH1 *hist3 = new TH1F("h3","L5+L6",maxtracklet+1,-0.5,maxtracklet+0.5);


 ifstream in("tracklets.txt");

 int count=0;

 while (in.good()){

   int layer,npairs;
  

   in >>layer>>npairs;

   if (!in.good()) continue;

   //cout <<layer<<" "<<r<<endl;

   if (layer==1) hist1->Fill(npairs);
   if (layer==3) hist2->Fill(npairs);
   if (layer==5) hist3->Fill(npairs);

   count++;

 }

 cout << "Processed: "<<count<<" events"<<endl;


 c1->cd(1);
 gPad->SetLogy();
 h1->GetYaxis()->SetTitle("Entries");
 h1->GetXaxis()->SetTitle("Tracklets per VM pair");
 h1->Draw();

 c1->cd(2);
 h1->GetYaxis()->SetTitle("Entries");
 h1->GetXaxis()->SetTitle("Tracklets per VM pair");
 h1->Draw();

 
 c1->cd(3);
 gPad->SetLogy();
 h2->GetYaxis()->SetTitle("Entries");
 h2->GetXaxis()->SetTitle("Tracklets per VM pair");
 h2->Draw();

 c1->cd(4);
 h2->GetYaxis()->SetTitle("Entries");
 h2->GetXaxis()->SetTitle("Tracklets per VM pair");
 h2->Draw();


 c1->cd(5);
 gPad->SetLogy();
 h3->GetYaxis()->SetTitle("Entries");
 h3->GetXaxis()->SetTitle("Tracklets per VM pair");
 h3->Draw();

 c1->cd(6);
 h3->GetYaxis()->SetTitle("Entries");
 h3->GetXaxis()->SetTitle("Tracklets per VM pair");
 h3->Draw();


 c1->Print("trackletsperVM.pdf");
 c1->Print("trackletsperVM.png");

}

