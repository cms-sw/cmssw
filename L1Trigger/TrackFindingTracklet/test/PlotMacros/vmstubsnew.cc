#include "TMath.h"
#include "TRint.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TLorentzVector.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TF1.h"
#include "TGaxis.h"
#include <fstream>
#include <iostream>
#include "TMath.h"
#include <string>
#include <map>



void vmstubsnew(){

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




 TCanvas* c1 = new TCanvas("c1","Track performance",200,10,700,800);
 c1->Divide(2,3);
 c1->SetFillColor(0);
 c1->SetGrid();

 
 TH1 *hist1 = new TH1F("h1","Stubs per VM L1",30,0.0,30.0);
 TH1 *hist2 = new TH1F("h2","Stubs per VM L2",30,0.0,30.0);
 TH1 *hist3 = new TH1F("h3","Stubs per VM L3",30,0.0,30.0);
 TH1 *hist4 = new TH1F("h4","Stubs per VM L4",30,0.0,30.0);
 TH1 *hist5 = new TH1F("h5","Stubs per VM L5",30,0.0,30.0);
 TH1 *hist6 = new TH1F("h6","Stubs per VM L6",30,0.0,30.0);


 ifstream in("newvmoccupancy.txt");

 int count=0;

 while (in.good()){

   int layer,nstub;

   in >>layer>>nstub;

   //cout <<layer<<" "<<r<<endl;

   if (layer==1) hist1->Fill(nstub);
   if (layer==2) hist2->Fill(nstub);
   if (layer==3) hist3->Fill(nstub);
   if (layer==4) hist4->Fill(nstub);
   if (layer==5) hist5->Fill(nstub);
   if (layer==6) hist6->Fill(nstub);

   count++;

 }

 cout << "Processed: "<<count<<" events"<<endl;

 c1->cd(1);
 gPad->SetLogy();
 TF1* f1 = new TF1("f1","[0]*TMath::Power([1],x)*(TMath::Exp(-[1]))/TMath::Gamma(x+1)", 0, 30);
 f1->SetParameters(1.0, 1.0); 
 hist1->Fit("f1", "R"); 
 hist1->Draw();

 c1->cd(2);
 gPad->SetLogy();
 TF1* f2 = new TF1("f2","[0]*TMath::Power([1],x)*(TMath::Exp(-[1]))/TMath::Gamma(x+1)", 0, 30);
 f2->SetParameters(1.0, 1.0); 
 hist2->Fit("f2", "R"); 
 hist2->Draw();

 c1->cd(3);
 gPad->SetLogy();
 TF1* f3 = new TF1("f3","[0]*TMath::Power([1],x)*(TMath::Exp(-[1]))/TMath::Gamma(x+1)", 0, 30);
 f3->SetParameters(1.0, 1.0); 
 hist3->Fit("f3", "R"); 
 hist3->Draw();

 c1->cd(4);
 gPad->SetLogy();
 TF1* f4 = new TF1("f4","[0]*TMath::Power([1],x)*(TMath::Exp(-[1]))/TMath::Gamma(x+1)", 0, 30);
 f4->SetParameters(1.0, 1.0); 
 hist4->Fit("f4", "R"); 
 hist4->Draw();

 c1->cd(5);
 gPad->SetLogy();
 TF1* f5 = new TF1("f5","[0]*TMath::Power([1],x)*(TMath::Exp(-[1]))/TMath::Gamma(x+1)", 0, 30);
 f5->SetParameters(1.0, 1.0); 
 hist5->Fit("f5", "R"); 
 hist5->Draw();

 c1->cd(6);
 gPad->SetLogy();
 TF1* f6 = new TF1("f6","[0]*TMath::Power([1],x)*(TMath::Exp(-[1]))/TMath::Gamma(x+1)", 0, 30);
 f6->SetParameters(1.0, 1.0); 
 hist6->Fit("f6", "R"); 
 hist6->Draw();


 c1->Print("vmstubsnew.png");
 c1->Print("vmstubsnew.pdf");

}

