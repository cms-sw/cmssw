#include "TMath.h"
#include "TRint.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TLorentzVector.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TF1.h"
#include "TLatex.h"
#include "TPaveStats.h"
#include "TLegend.h"
#include "TGaxis.h"
#include <fstream>
#include <iostream>
#include "TMath.h"
#include <string>
#include <map>


void mySmallText(Double_t x,Double_t y,Color_t color,const char *text) {
  Double_t tsize=0.044;
  TLatex l;
  l.SetTextSize(tsize); 
  l.SetNDC();
  l.SetTextColor(color);
  l.DrawLatex(x,y,text);
}


void trackres(){
//
// To see the output of this macro, click here.



//


gROOT->Reset();

gROOT->SetStyle("Plain");

gStyle->SetCanvasColor(kWhite);

gStyle->SetCanvasBorderMode(0);     // turn off canvas borders
gStyle->SetPadBorderMode(0);
//gStyle->SetOptStat(0);
gStyle->SetOptTitle(0);

  // For publishing:
  gStyle->SetLineWidth(1);
  gStyle->SetTextSize(1.1);
  gStyle->SetLabelSize(0.06,"xy");
  gStyle->SetTitleSize(0.06,"xy");
  gStyle->SetTitleOffset(1.0,"x");
  gStyle->SetTitleOffset(1.6,"y");
  gStyle->SetPadTopMargin(0.1);
  gStyle->SetPadRightMargin(0.06);
  gStyle->SetPadBottomMargin(0.16);
  gStyle->SetPadLeftMargin(0.18);




 TCanvas* c1 = new TCanvas("c1","Track performance",250,50,600,700);
 c1->Divide(2,2);
 c1->SetFillColor(0);
 c1->SetGrid();

 TCanvas* c2 = new TCanvas("c2","Track performance",250,50,600,700);
 c2->Divide(2,2);
 c2->SetFillColor(0);
 c2->SetGrid();

 TCanvas* c3 = new TCanvas("c3","Track performance",250,50,600,700);
 c3->Divide(2,2);
 c3->SetFillColor(0);
 c3->SetGrid();

 TCanvas* c4 = new TCanvas("c4","Track performance",250,40,600,650);
 c4->Divide(2,2);
 c4->SetFillColor(0);
 c4->SetGrid();

 TCanvas* c5 = new TCanvas("c5","Track performance",250,50,600,700);
 c5->Divide(2,2);
 c5->SetFillColor(0);
 c5->SetGrid();

 TCanvas* c6 = new TCanvas("c6","Track performance",250,50,600,700);
 c6->Divide(2,2);
 c6->SetFillColor(0);
 c6->SetGrid();


 double ptmax=10.0;
 double phimax=4.0;
 double detamax=40.0;
 double dz0max=2.0;

 TH1 *hist01 = new TH1F("h01","(ptfit-ptgen)/ptgen",30,-ptmax,ptmax);
 TH1 *hist02 = new TH1F("h02","phi0fit-phi0gen",20,-phimax,phimax);
 TH1 *hist03 = new TH1F("h03","etafit-etagen",50,-detamax,detamax);
 TH1 *hist04 = new TH1F("h04","z0fit-z0gen",50,-dz0max,dz0max);

 TH1 *hist11 = new TH1F("h11","(ptfitexact-ptgen)/ptgen",30,-ptmax,ptmax);
 TH1 *hist12 = new TH1F("h12","phi0fitexact-phi0gen",20,-phimax,phimax);
 TH1 *hist13 = new TH1F("h13","etafitexact-etagen",50,-detamax,detamax);
 TH1 *hist14 = new TH1F("h14","z0fitexact-z0gen",50,-dz0max,dz0max);

 TH1 *hist21 = new TH1F("h21","(iptfit-ptgen)/ptgen",30,-ptmax,ptmax);
 TH1 *hist22 = new TH1F("h22","iphi0fit-phi0gen",20,-phimax,phimax);
 TH1 *hist23 = new TH1F("h23","ietafit-etagen",50,-detamax,detamax);
 TH1 *hist24 = new TH1F("h24","iz0fit-z0gen",50,-dz0max,dz0max);

 TH1 *hist31 = new TH1F("h31","(ipt-ptgen)/ptgen",30,-ptmax,ptmax);
 TH1 *hist32 = new TH1F("h32","iphi0-phi0gen",20,-phimax,phimax);
 TH1 *hist33 = new TH1F("h33","ieta-etagen",50,-detamax,detamax);
 TH1 *hist34 = new TH1F("h34","iz0-z0gen",50,-dz0max,dz0max);

 TH1 *hist41 = new TH1F("h41","(ptapprox-ptgen)/ptgen",30,-ptmax,ptmax);
 TH1 *hist42 = new TH1F("h42","phi0approx-phi0gen",20,-phimax,phimax);
 TH1 *hist43 = new TH1F("h43","etaapprox-etagen",50,-detamax,detamax);
 TH1 *hist44 = new TH1F("h44","z0approx-z0gen",50,-dz0max,dz0max);

 TH1 *hist51 = new TH1F("h51","(ptapprox-ipt)/ptapprox",30,-ptmax,ptmax);
 TH1 *hist52 = new TH1F("h52","phi0approx-iphi0",20,-phimax,phimax);
 TH1 *hist53 = new TH1F("h53","etaapprox-ieta",50,-detamax*0.1,detamax*0.1);
 TH1 *hist54 = new TH1F("h54","z0approx-iz0",50,-dz0max*0.1,dz0max*0.1);

 TH1 *hist61 = new TH1F("h61","(ptapprox-ptfit)/ptgen",30,-ptmax,ptmax);
 TH1 *hist62 = new TH1F("h62","phi0approx-phi0fit",20,-phimax,phimax);
 TH1 *hist63 = new TH1F("h63","etaapprox-etafit",50,-detamax,detamax);
 TH1 *hist64 = new TH1F("h64","z0approx-z0fit",50,-dz0max,dz0max);


 TH1 *hist101 = new TH1F("h101","iphi0fit",20,0.0,2*3.141592);
 TH1 *hist102 = new TH1F("h102","iphi0fit",20,0.0,2*3.141592/28.0);


 ifstream in("trackres.txt");

 int count=0;

 while (in.good()){

   int layer,nmatch;
   double ptgen,phi0gen,etagen,z0gen;
   double ptfit,phi0fit,etafit,z0fit;
   double ptfitexact,phi0fitexact,etafitexact,z0fitexact;
   double iptfit,iphi0fit,ietafit,iz0fit;
   double ipt,iphi0,ieta,iz0;
   double ptapprox,phi0approx,etaapprox,z0approx;

   in >>layer>>nmatch
      >>ptgen>>phi0gen>>etagen>>z0gen
      >>ptfit>>phi0fit>>etafit>>z0fit
      >>ptfitexact>>phi0fitexact>>etafitexact>>z0fitexact
      >>iptfit>>iphi0fit>>ietafit>>iz0fit
      >>ipt>>iphi0>>ieta>>iz0
      >>ptapprox>>phi0approx>>etaapprox>>z0approx;


   //if (fabs(z0approx-z0fit)>1.0) continue;

   if (fabs(etagen)>2.4) continue;

   //if (layer!=1) continue;
   //if (nmatch<2) continue;


   //cout << "layer: "<<layer<<" "<<nmatch<<" "<<ptgen<<" "<<iptfit<<endl;

   ptgen=fabs(ptgen);

   //if (ptgen<5.0) continue;
   
   //cout << " pt "<<ptapprox<<" "<<ipt<<endl;

 
   if (iphi0fit<0.0) iphi0fit+=2*3.14159264;
   if (iphi0<0.0) iphi0+=2*3.14159264;
   if (iphi0fit>2*3.141592) iphi0fit-=2*3.14159264;

   int niphi0=28*(iphi0fit/(2*3.141592));

   double diphi=iphi0fit-niphi0*2*3.141592/28;

   hist101->Fill(iphi0fit);
   hist102->Fill(diphi);

   if (phi0gen<0.0) phi0gen+=2*3.14159264;
   
   //if (ptgen>30) continue;

   //cout << "phi0approx phi0gen "<<phi0approx<<" "<<phi0gen<<endl;
   //cout << "iphi0 phi0gen "<<iphi0<<" "<<phi0gen<<endl;

   hist01->Fill(100*(fabs(ptfit)-ptgen)/ptgen);
   hist02->Fill(asin(sin(phi0fit-phi0gen))*1000);
   hist03->Fill((etafit-etagen)*1000);
   hist04->Fill(z0fit-z0gen);

   hist11->Fill(100*(fabs(ptfitexact)-ptgen)/ptgen);
   hist12->Fill((phi0fitexact-phi0gen)*1000);
   hist13->Fill((etafitexact-etagen)*1000);
   hist14->Fill(z0fitexact-z0gen);

   if (fabs(100*(fabs(iptfit)-ptgen)/ptgen)>10.0) cout << "deltapt/pt : "<<100*((fabs(iptfit)-ptgen)/ptgen)<<" "<<iptfit<<" "<<ptgen<<" "<<etagen<<endl;

   hist21->Fill(100*(fabs(iptfit)-ptgen)/ptgen);
   hist22->Fill((iphi0fit-phi0gen)*1000);
   hist23->Fill((ietafit-etagen)*1000);
   hist24->Fill(iz0fit-z0gen);

   //cout << "iz0fit-z0gen = "<<iz0fit-z0gen<<endl;

   hist31->Fill(100*(fabs(ipt)-ptgen)/ptgen);
   hist32->Fill((iphi0-phi0gen)*1000);
   hist33->Fill((ieta-etagen)*1000);
   hist34->Fill(iz0-z0gen);

   hist41->Fill(100*(fabs(ptapprox)-ptgen)/ptgen);
   hist42->Fill(asin(sin(phi0approx-phi0gen))*1000);
   hist43->Fill((etaapprox-etagen)*1000);
   hist44->Fill(z0approx-z0gen);

   hist51->Fill(100*(fabs(ptapprox)-fabs(ipt))/ptapprox);
   hist52->Fill((phi0approx-iphi0)*1000);
   hist53->Fill((etaapprox-ieta)*1000);
   hist54->Fill(z0approx-iz0);

   hist61->Fill(100*(fabs(ipt)-fabs(ptapprox))/ptgen);
   hist62->Fill((iphi0-phi0approx)*1000);
   hist63->Fill((ieta-etaapprox)*1000);
   //cout << "ieta etafit : "<<etaapprox<<" "<<etaapprox<<endl;
   hist64->Fill(iz0-z0approx);
   
   //cout << "z0gen z0fitexact z0approx iz0 iz0fit : "<<z0gen<<" "
   //	<<z0approx<<" "
   //	<<z0fitexact<<" "
   //	<<iz0<<" "
   //	<<iz0fit<<endl;

   count++;

 }

 cout << "Processed: "<<count<<" events"<<endl;

 c1->cd(1);
 hist01->Draw();

 c1->cd(2);
 hist02->Draw();

 c1->cd(3);
 hist03->Draw();

 c1->cd(4);
 hist04->Draw();

 c1->Print("trackres1.pdf");
 c1->Print("trackres1.png");

 c2->cd(1);
 hist11->Draw();

 c2->cd(2);
 hist12->Draw();

 c2->cd(3);
 hist13->Draw();

 c2->cd(4);
 hist14->Draw();

 c2->Print("trackres2.pdf");
 c2->Print("trackres2.png");

 c3->cd(1);
 hist01->SetLineColor(kBlue);
 hist01->GetXaxis()->SetTitle("#sigma(p_{T})/p_{T} [%]");
 hist01->GetYaxis()->SetTitle("Tracks");
 //hist01->SetMaximum(1500);
 hist01->Draw();
 hist21->SetLineColor(kRed);
 hist21->Draw("same");
 TLegend* leg1 = new TLegend(0.18,0.75,0.45,0.9);
 leg1->AddEntry(hist01,"Floating point","l");
 leg1->AddEntry(hist21,"Integer emulation","l");
 leg1->Draw();
 mySmallText(0.18,0.91,1,"CMS Preliminary Simulation, HL-LHC"); 


 c3->cd(2);
 hist02->SetLineColor(kBlue);
 hist02->GetXaxis()->SetTitle("#sigma(#phi_{0}) [mrad]");
 hist02->GetYaxis()->SetTitle("Tracks");
 //hist02->SetMaximum(1500);
 hist02->Draw();
 hist22->SetLineColor(kRed);
 hist22->Draw("same");
 TLegend* leg2 = new TLegend(0.18,0.75,0.45,0.9);
 leg2->AddEntry(hist02,"Floating point","l");
 leg2->AddEntry(hist22,"Integer emulation","l");
 leg2->Draw();
 mySmallText(0.18,0.91,1,"CMS Preliminary Simulation, HL-LHC"); 

 c3->cd(3);
 hist03->SetLineColor(kBlue);
 hist03->GetXaxis()->SetTitle("#sigma(#eta) [10^{-3}]");
 hist03->GetYaxis()->SetTitle("Tracks");
 //hist03->SetMaximum(1200);
 hist03->Draw();
 hist23->SetLineColor(kRed);
 hist23->Draw("same");
 TLegend* leg3 = new TLegend(0.18,0.75,0.45,0.9);
 leg3->AddEntry(hist03,"Floating point","l");
 leg3->AddEntry(hist23,"Integer emulation","l");
 leg3->Draw();
 mySmallText(0.18,0.91,1,"CMS Preliminary Simulation, HL-LHC"); 

 c3->cd(4);
 hist04->SetLineColor(kBlue);
 hist04->GetXaxis()->SetTitle("#sigma(z_{0}) [cm]");
 hist04->GetYaxis()->SetTitle("Tracks");
 //hist04->SetMaximum(1200);
 hist04->Draw();

 gPad->Update();
 TPaveStats* stats =(TPaveStats*)hist04->FindObject("stats");
 cout << "stats:"<<stats<<endl;
 stats->SetName("h1stats");
 stats->SetY1NDC(.7);
 stats->SetY2NDC(.9);
 stats->SetTextColor(kBlue);
 
 hist24->SetLineColor(kRed);
 hist24->Draw("sames");
 gPad->Update();
 TPaveStats* stats2 =(TPaveStats*)hist24->FindObject("stats");
 cout << "stats2:"<<stats2<<endl;
 stats2->SetName("h1stats2");
 stats2->SetY1NDC(.5);
 stats2->SetY2NDC(.7);
 stats2->SetTextColor(kRed);
 
 TLegend* leg4 = new TLegend(0.18,0.75,0.45,0.9);
 leg4->AddEntry(hist04,"Floating point","l");
 leg4->AddEntry(hist24,"Integer emulation","l");
 leg4->Draw();
 mySmallText(0.18,0.91,1,"CMS Preliminary Simulation, HL-LHC"); 

 c3->Print("trackres.pdf");
 c3->Print("trackres.png");

 c4->cd(1);
 hist101->SetMinimum(0.0);
 hist101->GetYaxis()->SetTitle("Tracks");
 hist101->GetXaxis()->SetTitle("#phi [rad]");
 hist101->Draw();

 c4->cd(2);
 hist102->SetMinimum(0.0);
 hist102->GetXaxis()->SetNdivisions(505);
 hist102->GetYaxis()->SetTitle("Tracks");
 hist102->GetXaxis()->SetTitle("Local Sector #phi [rad]");
 hist102->Draw();

 c4->Print("trackres4.pdf");
 c4->Print("trackres4.png");


 c5->cd(1);
 hist41->SetLineColor(kBlue);
 hist41->GetXaxis()->SetTitle("Tracklet #sigma(p_{T})/p_{T} [%]");
 hist41->GetYaxis()->SetTitle("Tracks");
 hist41->Draw();
 hist31->SetLineColor(kRed);
 hist31->Draw("same");
 TLegend* leg41 = new TLegend(0.18,0.75,0.45,0.9);
 leg41->AddEntry(hist01,"Floating point","l");
 leg41->AddEntry(hist21,"Integer emulation","l");
 leg41->Draw();
 mySmallText(0.18,0.91,1,"CMS Preliminary Simulation, HL-LHC"); 


 c5->cd(2);
 hist42->SetLineColor(kBlue);
 hist42->GetXaxis()->SetTitle("Tracklet #sigma(#phi_{0}) [mrad]");
 hist42->GetYaxis()->SetTitle("Tracks");
 hist42->Draw();
 hist32->SetLineColor(kRed);
 hist32->Draw("same");
 TLegend* leg42 = new TLegend(0.18,0.75,0.45,0.9);
 leg42->AddEntry(hist02,"Floating point","l");
 leg42->AddEntry(hist22,"Integer emulation","l");
 leg42->Draw();
 mySmallText(0.18,0.91,1,"CMS Preliminary Simulation, HL-LHC"); 


 c5->cd(3);
 hist43->SetLineColor(kBlue);
 hist43->GetXaxis()->SetTitle("Tracklet #sigma(#eta) [10^{-3}]");
 hist43->GetYaxis()->SetTitle("Tracks");
 hist43->Draw();
 hist33->SetLineColor(kRed);
 hist33->Draw("same");
 TLegend* leg43 = new TLegend(0.18,0.75,0.45,0.9);
 leg43->AddEntry(hist03,"Floating point","l");
 leg43->AddEntry(hist23,"Integer emulation","l");
 leg43->Draw();
 mySmallText(0.18,0.91,1,"CMS Preliminary Simulation, HL-LHC"); 


 c5->cd(4);
 hist44->SetLineColor(kBlue);
 hist44->GetXaxis()->SetTitle("Tracklet #sigma(z_{0}) [cm]");
 hist44->GetYaxis()->SetTitle("Tracks");
 hist44->Draw();
 hist34->SetLineColor(kRed);
 hist34->Draw("same");
 TLegend* leg44 = new TLegend(0.18,0.75,0.45,0.9);
 leg44->AddEntry(hist04,"Floating point","l");
 leg44->AddEntry(hist24,"Integer emulation","l");
 leg44->Draw();
 mySmallText(0.18,0.91,1,"CMS Preliminary Simulation, HL-LHC"); 


 c5->Print("trackletres.pdf");
 c5->Print("trackletres.png");


 c6->cd(1);
 hist61->Draw();
 c6->cd(2);
 hist62->Draw();
 c6->cd(3);
 hist63->Draw();
 c6->cd(4);
 hist64->Draw();

 c6->Print("diff.pdf");
 c6->Print("diff.png");

 
}


