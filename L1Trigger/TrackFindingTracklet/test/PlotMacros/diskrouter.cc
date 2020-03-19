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



void diskrouter(){
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




 TCanvas* c1 = new TCanvas("c1","Track performance",200,10,800,300);
 c1->Divide(2,1);
 c1->SetFillColor(0);
 c1->SetGrid();

 double max=100.0;
 
 TH1 *hist1 = new TH1F("h1","Disk Router Inner",max+1,-0.5,max+0.5);
 TH1 *hist2 = new TH1F("h2","Disk Router Outer",max+1,-0.5,max+0.5);

 ifstream in("diskrouter.txt");

 int count=0;

 while (in.good()){

   TString name;
   int projs;
  
   in >>name>>projs;

   if (!in.good()) continue;
   //cout << name <<" "<< countall <<" "<< countpass << endl;

   if (projs>max) projs=max;

   if (name.Contains("_D5")||
       name.Contains("_D7")) hist1->Fill(projs);
   if (name.Contains("_D6")||
       name.Contains("_D8")) hist2->Fill(projs);

   count++;

 }

 cout << "count = "<<count<<endl;

 c1->cd(1);
 gPad->SetLogy();
 hist1->Draw();

 c1->cd(2);
 gPad->SetLogy();
 hist2->Draw();

 c1->Print("diskrouter.pdf");

}
