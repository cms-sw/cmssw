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



void trackletpars(){
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




 TCanvas* c1 = new TCanvas("c1","Track performance",200,100,400,600);
 c1->Divide(2,4);
 c1->SetFillColor(0);
 c1->SetGrid();

 TCanvas* c2 = new TCanvas("c2","Track performance",220,100,400,600);
 c2->Divide(2,2);
 c2->SetFillColor(0);
 c2->SetGrid();

 TCanvas* c3 = new TCanvas("c3","Track performance",240,100,400,600);
 c3->Divide(2,2);
 c3->SetFillColor(0);
 c3->SetGrid();

 
 TH1 *hist01 = new TH1F("h01","rinv approx",50,-100.0,100.0);
 TH1 *hist02 = new TH1F("h02","rinv int",50,-100.0,100.0);
 TH1 *hist03 = new TH1F("h03","pt",50,0.0,150.0);

 TH1 *hist11 = new TH1F("h11","phi0 approx",50,-2.0,2.0);
 TH1 *hist12 = new TH1F("h12","phi0 int",50,-2.0,2.0);
 TH1 *hist13 = new TH1F("h13","phi0",50,-3.141592,3.141592);

 TH1 *hist21 = new TH1F("h21","t approx",50,-100.0,100.0);
 TH1 *hist22 = new TH1F("h22","t int",50,-100.0,100.0);
 TH1 *hist23 = new TH1F("h23","eta",50,-3.0,3.0);

 TH1 *hist31 = new TH1F("h31","z0 approx",50,-10.0,10.0);
 TH1 *hist32 = new TH1F("h32","z0 int",50,-10.0,10.0);
 TH1 *hist33 = new TH1F("h33","z0 int",50,-25.0,25.0);


 ifstream in("trackletpars.txt");

 int count=0;

 while (in.good()){

   string tmp;
   int layer;
   double  rinv,rinvapprox,irinv;
   double  phi0,phi0approx,iphi0;
   double  t,tapprox,it;
   double  z0,z0approx,iz0; 


   in >>tmp>>layer>>rinv>>rinvapprox>>irinv>>phi0>>phi0approx>>iphi0
      >>t>>tapprox>>it>>z0>>z0approx>>iz0;

   //if (fabs(rinv)>0.00057) continue;
   //if (fabs(t)>1.0) continue;

   if (layer!=1) continue;

   //cout <<tmp<<" "<<layer<<" "<<rinv<<endl;

   hist01->Fill((rinvapprox-rinv)*1e6);
   hist02->Fill((irinv-rinv)*1e6);
   hist03->Fill((2.0*0.0057)/fabs(irinv));

   hist11->Fill((phi0approx-phi0)*1e3);
   hist12->Fill((iphi0-phi0)*1e3);
   hist13->Fill(iphi0);

   //if (fabs(iphi0-phi0)>4e-5) cout << "iphi0-phi0 rinv pt: "<<iphi0-phi0
   //				  <<" "<<rinv<<" "<<0.3*3.8/(100*rinv)<<endl; 

   hist21->Fill((tapprox-t)*1e3);
   hist22->Fill((it-t)*1e3);
   hist23->Fill(log(it+sqrt(1.0+it*it)));

   hist31->Fill(z0approx-z0);
   hist32->Fill(iz0-z0);
   hist33->Fill(iz0);


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

 c1->Print("trackpars.pdf");
 c1->Print("trackpars.png");


 c2->cd(1);
 hist02->GetXaxis()->SetTitle("#Delta#rho^{-1} [10^{-6} cm^{-1}]");
 hist02->Draw();
 hist01->Draw("SAME");


 c2->cd(2);
 hist12->GetXaxis()->SetTitle("#Delta#phi_{0} [mrad]");
 hist12->Draw();
 hist11->Draw("SAME");

 c2->cd(3);
 hist22->GetXaxis()->SetTitle("#Delta t [10^{-3}]");
 hist22->Draw();
 hist21->Draw("SAME");

 c2->cd(4);
 hist32->GetXaxis()->SetTitle("#Delta z_{0} [cm]");
 hist32->Draw();
 hist31->Draw("SAME");


 c2->Print("trackletpars.pdf");
 c2->Print("trackletpars.png");

 c3->cd(1);
 hist03->GetXaxis()->SetTitle("P_t GeV]");
 hist03->Draw();

 c3->cd(2);
 hist13->GetXaxis()->SetTitle("#phi_{0} [rad]");
 hist13->Draw();

 c3->cd(3);
 hist23->GetXaxis()->SetTitle("#eta");
 hist23->Draw();

 c3->cd(4);
 hist33->GetXaxis()->SetTitle("z_{0} [cm]");
 hist33->Draw();

 c3->Print("trackletpars.pdf");
 c3->Print("trackletpars.png");

}

