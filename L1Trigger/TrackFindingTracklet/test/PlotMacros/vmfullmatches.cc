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
 c1->Divide(2,3);
 c1->SetFillColor(0);
 c1->SetGrid();

 c2 = new TCanvas("c2","Track performance",200,10,700,800);
 c2->Divide(2,3);
 c2->SetFillColor(0);
 c2->SetGrid();

 
 TH1 *hist1 = new TH1F("h1","Full matches per VM L1",15,0.0,25.0);
 TH1 *hist2 = new TH1F("h2","Full matches per VM L2",15,0.0,25.0);
 TH1 *hist3 = new TH1F("h3","Full matches per VM L3",15,0.0,25.0);
 TH1 *hist4 = new TH1F("h4","Full matches per VM L4",15,0.0,25.0);
 TH1 *hist5 = new TH1F("h5","Full matches per VM L5",15,0.0,25.0);
 TH1 *hist6 = new TH1F("h6","Full matches per VM L6",15,0.0,25.0);

 TH1 *hist11 = new TH1F("h11","Full matches L1",40,0.0,80.0);
 TH1 *hist12 = new TH1F("h12","Full matches L2",40,0.0,80.0);
 TH1 *hist13 = new TH1F("h13","Full matches L3",40,0.0,80.0);
 TH1 *hist14 = new TH1F("h14","Full matches L4",40,0.0,80.0);
 TH1 *hist15 = new TH1F("h15","Full matches L5",40,0.0,80.0);
 TH1 *hist16 = new TH1F("h16","Full matches L6",40,0.0,80.0);


 ifstream in("vmfullmatch.txt");

 int count=0;
  
 int seedlayerold=3;
 int layerold=1;
 int summatch=0;

 while (in.good()){

   int layer,seedlayer,nmatch,i,j;
  
   in >>layer>>seedlayer>>i>>j>>nmatch;

   if (!in.good()) continue;

   if (layer==layerold&&seedlayer==seedlayerold){
     summatch+=nmatch;
   } else {
     if (layerold==1) hist11->Fill(summatch);
     if (layerold==2) hist12->Fill(summatch);
     if (layerold==3) hist13->Fill(summatch);
     if (layerold==4) hist14->Fill(summatch);
     if (layerold==5) hist15->Fill(summatch);
     if (layerold==6) hist16->Fill(summatch);
     summatch=nmatch;
     seedlayerold=seedlayer;
     layerold=layer;
   }

   //cout <<layer<<" "<<r<<endl;

   if (layer==1) hist1->Fill(nmatch);
   if (layer==2) hist2->Fill(nmatch);
   if (layer==3) hist3->Fill(nmatch);
   if (layer==4) hist4->Fill(nmatch);
   if (layer==5) hist5->Fill(nmatch);
   if (layer==6) hist6->Fill(nmatch);

   count++;

 }

 if (layerold==1) hist11->Fill(summatch);
 if (layerold==2) hist12->Fill(summatch);
 if (layerold==3) hist13->Fill(summatch);
 if (layerold==4) hist14->Fill(summatch);
 if (layerold==5) hist15->Fill(summatch);
 if (layerold==6) hist16->Fill(summatch);


 cout << "Processed: "<<count<<" events"<<endl;

 c1->cd(1);
 gPad->SetLogy();
 h1->Draw();

 c1->cd(2);
 gPad->SetLogy();
 h2->Draw();

 c1->cd(3);
 gPad->SetLogy();
 h3->Draw();

 c1->cd(4);
 gPad->SetLogy();
 h4->Draw();

 c1->cd(5);
 gPad->SetLogy();
 h5->Draw();

 c1->cd(6);
 gPad->SetLogy();
 h6->Draw();


 c1->Print("vmfullmatches.png");
 c1->Print("vmfullmatches.pdf");

 c2->cd(1);
 gPad->SetLogy();
 h11->Draw();

 c2->cd(2);
 gPad->SetLogy();
 h12->Draw();

 c2->cd(3);
 gPad->SetLogy();
 h13->Draw();

 c2->cd(4);
 gPad->SetLogy();
 h14->Draw();

 c2->cd(5);
 gPad->SetLogy();
 h15->Draw();

 c2->cd(6);
 gPad->SetLogy();
 h16->Draw();


 c2->Print("fullmatches.png");
 c2->Print("fullmatches.pdf");

}

