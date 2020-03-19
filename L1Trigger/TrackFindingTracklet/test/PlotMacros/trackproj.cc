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



void trackproj(){
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




 TCanvas* c1 = new TCanvas("c1","Track performance",200,10,600,1000);
 c1->Divide(2,4);
 c1->SetFillColor(0);
 c1->SetGrid();

 TCanvas* c2 = new TCanvas("c2","Track performance",200,10,600,1000);
 c2->Divide(2,2);
 c2->SetFillColor(0);
 c2->SetGrid();

 
 TH1 *hist01 = new TH1F("h01","phiproj approx",50,-0.05,0.05);
 TH1 *hist02 = new TH1F("h02","phiproj int",50,-0.05,0.05);

 TH1 *hist11 = new TH1F("h11","rproj approx",50,-2.0,2.0);
 TH1 *hist12 = new TH1F("h12","rproj int",50,-2.0,2.0);

 TH1 *hist21 = new TH1F("h21","phider approx",50,-0.01,0.01);
 TH1 *hist22 = new TH1F("h22","phider int",50,-0.01,0.01);

 TH1 *hist31 = new TH1F("h31","zder approx",50,-0.5,0.5);
 TH1 *hist32 = new TH1F("h32","zder int",50,-0.5,0.5);


 ifstream in("trackproj.txt");

 int count=0;

 while (in.good()){

   string tmp;
   int layer;
   double  rproj;
   double  phiproj,phiprojapprox,iphiproj;
   double  zproj,zprojapprox,izproj;
   double  phider,phiderapprox,iphider;
   double  zder,zderapprox,izder;


   in >>tmp>>tmp>>layer>>rproj
      >>phiproj>>phiprojapprox>>iphiproj
      >>zproj>>zprojapprox>>izproj
      >>phider>>phiderapprox>>iphider
      >>zder>>zderapprox>>izder;

   if (layer!=1) continue;
   if (fabs(rproj-260.0)>10.0) continue;

   cout <<tmp<<endl;
   cout << zproj << " " << zprojapprox << " " << izproj << endl;

   hist01->Fill(rproj*(phiprojapprox-phiproj));
   hist02->Fill(rproj*(iphiproj-phiproj));

   hist11->Fill(zprojapprox-zproj);
   hist12->Fill(izproj-zproj);

   hist21->Fill(phiderapprox-phider);
   hist22->Fill(iphider-phider);

   hist31->Fill(zderapprox-zder);
   hist32->Fill(izder-zder);



   count++;

 }

 cout << "Processed: "<<count<<" events"<<endl;

 c1->cd(1);
 hist01->Draw();

 c1->cd(2);
 hist02->Draw();

 c1->cd(3);
 hist11->Draw();

 c1->cd(4);
 hist12->Draw();

 c1->cd(5);
 hist21->Draw();

 c1->cd(6);
 hist22->Draw();

 c1->cd(7);
 hist31->Draw();

 c1->cd(8);
 hist32->Draw();

 c1->Print("trackproj.pdf");
 c1->Print("trackproj.png");



 c2->cd(1);
 hist02->GetXaxis()->SetTitle("r#Delta#phi_{proj} [cm]");
 hist02->Draw();

 c2->cd(2);
 hist12->GetXaxis()->SetTitle("#Delta z_{proj} [cm]");
 hist12->Draw();

 c2->cd(3);
 hist22->GetXaxis()->SetTitle("#partial_{r}#phi_{proj}");
 hist22->Draw();

 c2->cd(4);
 hist32->GetXaxis()->SetTitle("#partial_{r} z_{proj}");
 hist32->Draw();

 c2->Print("trackletproj.pdf");
 c2->Print("trackletproj.png");

}

