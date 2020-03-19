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
gStyle->SetOptStat(0);


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


 c1 = new TCanvas("c1","Track performance",200,10,1000,1100);
 c1->Divide(2,2);
 c1->SetFillColor(0);
 c1_4->SetGridy(10);
 c1_3->SetGridy(10);

 TString parttype="Muon";
 //TString parttype="Electron";
 //TString parttype="Pion";
 //TString parttype="Kaon";
 //TString parttype="Proton";

 int itype=-1;

 if (parttype=="Muon") itype=13;
 if (parttype=="Electron") itype=11;
 if (parttype=="Pion") itype=211;
 if (parttype=="Kaon") itype=321;
 if (parttype=="Proton") itype=2212;

 if (itype==-1) {
   cout << "Need to correctly specify parttype:"<<parttype<<endl;
 }

 TH1 *hist7 = new TH1F("h7","stubphi",50,0.0,0.3);

 hist7->GetXaxis()->SetTitle("#phi");
 hist7->GetYaxis()->SetTitle("Events");


 ifstream in("stubdisk.log");

 int count=0;
 int counteff=0;

 

 while (in.good()){

   count++;

   string tmp;

   int disk,iphi;
   double phi;

   in>>tmp>>phi;

   double phisector=phi;

   //while(phisector<0.0) phisector+=3.141592/14.0;
   //while(phisector>3.141592/14.0) phisector-=3.141592/14.0;

   hist7->Fill(phisector);

 }

 cout << "Processed: "<<count<<" events with <eff>="<<counteff*1.0/count<<endl;

 c1->cd(1);
 h7->Draw();

}

