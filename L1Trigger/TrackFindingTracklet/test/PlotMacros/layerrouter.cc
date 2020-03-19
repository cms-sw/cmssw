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



void layerrouter(){
//
// To see the output of this macro, click here.

//

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
 c1->Divide(2,2);
 c1->SetFillColor(0);
 c1->SetGrid();

 double max=100.0;
 
 TH1 *hist1 = new TH1F("h1","Layer Router D1",max+1,-0.5,max+0.5);
 TH1 *hist2 = new TH1F("h2","Layer Router D2",max+1,-0.5,max+0.5);
 TH1 *hist3 = new TH1F("h3","Layer Router D3",max+1,-0.5,max+0.5);
 TH1 *hist4 = new TH1F("h4","Layer Router D4",max+1,-0.5,max+0.5);

 ifstream in("layerrouter.txt");

 int count=0;

 while (in.good()){

   TString name;
   int projs;
  
   in >>name>>projs;

   if (!in.good()) continue;
   //cout << name <<" "<< countall <<" "<< countpass << endl;

   if (projs>max) projs=max;

   if (name.Contains("_D1")) hist1->Fill(projs);
   if (name.Contains("_D2")) hist2->Fill(projs);
   if (name.Contains("_D3")) hist3->Fill(projs);
   if (name.Contains("_D4")) hist4->Fill(projs);

   count++;

 }

 cout << "count = "<<count<<endl;

 c1->cd(1);
 gPad->SetLogy();
 hist1->Draw();

 c1->cd(2);
 gPad->SetLogy();
 hist2->Draw();

 c1->cd(3);
 gPad->SetLogy();
 hist3->Draw();

 c1->cd(4);
 gPad->SetLogy();
 hist4->Draw();

 c1->Print("layerrouter.pdf");

}
