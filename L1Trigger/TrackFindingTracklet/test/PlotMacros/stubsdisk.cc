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





 c4 = new TCanvas("c4","Occupancy",200,10,700,1000);
 c4->Divide(2,3);
 c4->SetFillColor(0);
 c4->SetGrid();

 
 TH1 *hist1 = new TH1F("h1","Number of stubs D1",50,0.0,150.0);
 TH1 *hist2 = new TH1F("h2","Number of stubs D2",50,0.0,150.0);
 TH1 *hist3 = new TH1F("h3","Number of stubs D3",50,0.0,150.0);
 TH1 *hist4 = new TH1F("h4","Number of stubs D4",50,0.0,150.0);
 TH1 *hist5 = new TH1F("h5","Number of stubs D5",50,0.0,150.0);


 ifstream in("stubsdisk.txt");

 int count=0;

 int disk,nf,nb;
  
 in >>disk>>nf>>nb;

 while (in.good()){

   if (disk==1) {
     hist1->Fill(nf);
     hist1->Fill(nb);
   }
   if (disk==2) {
     hist2->Fill(nf);
     hist2->Fill(nb);
   }
   if (disk==3) {
     hist3->Fill(nf);
     hist3->Fill(nb);
   }
   if (disk==4) {
     hist4->Fill(nf);
     hist4->Fill(nb);
   }
   if (disk==5) {
     hist5->Fill(nf);
     hist5->Fill(nb);
   }

   in >>disk>>nf>>nb;

   count++;

 }

 cout << "Processed: "<<count<<" events"<<endl;

 c4->cd(1);
 h1->Draw();

 c4->cd(2);
 h2->Draw();

 c4->cd(3);
 h3->Draw();

 c4->cd(4);
 h4->Draw();

 c4->cd(5);
 h5->Draw();

 c4->Print("stubsperdisk.png");
 c4->Print("stubsperdisk.pdf");


}

