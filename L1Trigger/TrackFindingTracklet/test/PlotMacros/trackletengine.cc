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
#include "plotHist.cc"


void trackletengine(){
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
  gStyle->SetLineWidth(1);
  gStyle->SetTextSize(1.1);
  gStyle->SetLabelSize(0.06,"xy");
  gStyle->SetTitleSize(0.06,"xy");
  gStyle->SetTitleOffset(1.2,"x");
  gStyle->SetTitleOffset(1.0,"y");
  gStyle->SetPadTopMargin(0.1);
  gStyle->SetPadRightMargin(0.1);
  gStyle->SetPadBottomMargin(0.16);
  gStyle->SetPadLeftMargin(0.12);

  
  int ncut1=72;
  int ncut2=108;

 TCanvas* c1 = new TCanvas("c1","Track performance",200,10,700,800);
 c1->Divide(2,2);
 c1->SetFillColor(0);
 c1->SetGrid();

 TCanvas* c2 = new TCanvas("c2","Track performance",200,10,700,800);
 c2->Divide(2,3);
 c2->SetFillColor(0);
 c2->SetGrid();

 TCanvas* c3 = new TCanvas("c3","Track performance",200,10,700,800);
 c3->Divide(2,3);
 c3->SetFillColor(0);
 c3->SetGrid();


 bool full=true;

 double max=128.0;
 
 TH1 *hist1 = new TH1F("h1","Number of stub pairs tried per VM pair",max+1,-0.5,max+0.5);
 TH1 *hist2 = new TH1F("h2","Number of stub pairs accepted per VM pair",max+1,-.5,max+0.5);

 TH1 *hist11 = new TH1F("h11","Number of stub pairs tried in L1L2 per VM pair",max+1,-0.5,max+0.5);
 TH1 *hist12 = new TH1F("h12","Number of stub pairs accepted in L1L2 per VM pair",max+1,-.5,max+0.5);

 TH1 *hist21 = new TH1F("h21","Number of stub pairs tried in L3L4 per VM pair",max+1,-0.5,max+0.5);
 TH1 *hist22 = new TH1F("h22","Number of stub pairs accepted in L3L4 per VM pair",max+1,-.5,max+0.5);

 TH1 *hist31 = new TH1F("h31","Number of stub pairs tried in L5L6 per VM pair",max+1,-0.5,max+0.5);
 TH1 *hist32 = new TH1F("h32","Number of stub pairs accepted in L5L6 per VM pair",max+1,-.5,max+0.5);

 TH1 *hist41 = new TH1F("h41","Number of stub pairs tried in D1D2 per VM pair",max+1,-0.5,max+0.5);
 TH1 *hist42 = new TH1F("h42","Number of stub pairs accepted in D1D2 per VM pair",max+1,-.5,max+0.5);

 TH1 *hist51 = new TH1F("h51","Number of stub pairs tried in D3D4 per VM pair",max+1,-0.5,max+0.5);
 TH1 *hist52 = new TH1F("h52","Number of stub pairs accepted in D3D4 per VM pair",max+1,-.5,max+0.5);

 TH1 *hist61 = new TH1F("h61","Number of stub pairs tried in L1D1 per VM pair",max+1,-0.5,max+0.5);
 TH1 *hist62 = new TH1F("h62","Number of stub pairs accepted in L1D1 per VM pair",max+1,-.5,max+0.5);



 ifstream in("trackletengine.txt");

 int count=0;

 std::map<TString, std::pair<TH1*,TH1*> > hists;

 while (in.good()){

   TString name;
   int countall,countpass;
  
   in >>name>>countall>>countpass;

   if (!in.good()) continue;
   //cout << name <<" "<< countall <<" "<< countpass << endl;

   if (countall>max) countall=max;
   if (countpass>max) countpass=max;

   if (name.Contains("_L1")&&name.Contains("_L2")) {
     hist11->Fill(countall);
     hist12->Fill(countpass);
   }

   if (name.Contains("_L3")&&name.Contains("_L4")) {
     hist21->Fill(countall);
     hist22->Fill(countpass);
   }

   if (name.Contains("_L5")&&name.Contains("_L6")) {
     hist31->Fill(countall);
     hist32->Fill(countpass);
   }

   if (name.Contains("_D1")&&name.Contains("_D2")) {
     hist41->Fill(countall);
     hist42->Fill(countpass);
   }

   if (name.Contains("_D3")&&name.Contains("_D4")) {
     hist51->Fill(countall);
     hist52->Fill(countpass);
   }

   if (name.Contains("_L1")&&name.Contains("_D1")) {
     hist61->Fill(countall);
     hist62->Fill(countpass);
   }


   hist1->Fill(countall);
   hist2->Fill(countpass);

   if (name.Contains("_D1")&&name.Contains("_D2")) {
   
     std::map<TString, std::pair<TH1*,TH1*> >::iterator it=hists.find(name);

     if (it==hists.end()) {
       TH1 *histall = new TH1F(name,name,max+1,-0.5,max+0.5);
       histall->Fill(countall);
       TH1 *histpass = new TH1F(name+"pass",name+"pass",max+1,-0.5,max+0.5);
       histpass->Fill(countpass);
       std::pair<TH1*,TH1*> tmp(histall,histpass);
       hists[name]=tmp;
     } else {
       hists[name].first->Fill(countall);
       hists[name].second->Fill(countpass);
     }
   }


   count++;

 }

 c1->cd(1);
 plotHist(hist1,0.05,ncut1,ncut2);

 c1->cd(2);
 plotHist(hist2,0.05,ncut1,ncut2);

 c1->cd(3);
 gPad->SetLogy();
 plotHist(hist1,0.05,ncut1,ncut2);

 c1->cd(4);
 gPad->SetLogy();
 plotHist(hist2,0.05,ncut1,ncut2);

 if (full) {

   int pages=0;

   std::map<TString, std::pair<TH1*,TH1*> >::iterator it=hists.begin();

   TCanvas* c=0;

   while(it!=hists.end()) {

     if (pages%4==0) {
     
       c = new TCanvas(it->first,"Track performance",200,50,600,700);
       c->Divide(2,2);
       c->SetFillColor(0);
       c->SetGrid();

     }

     c->cd(pages%4+1);
     gPad->SetLogy();
     plotHist(it->second.first,0.05,ncut1,ncut2);
     it->second.second->SetLineColor(kBlue);
     it->second.second->Draw("same");
     
     pages++;
     ++it;
   }
 }

 c2->cd(1);
 hist11->SetLineColor(kRed);
 plotHist(hist11,0.05,ncut1,ncut2);
 hist12->SetLineColor(kBlue);
 hist12->Draw("same");

 c2->cd(2);
 gPad->SetLogy();
 plotHist(hist11,0.05,ncut1,ncut2);
 hist12->Draw("same");

 c2->cd(3);
 hist21->SetLineColor(kRed);
 plotHist(hist21,0.05,ncut1,ncut2);
 hist22->SetLineColor(kBlue);
 hist22->Draw("same");

 c2->cd(4);
 gPad->SetLogy();
 plotHist(hist21,0.05,ncut1,ncut2);
 hist22->Draw("same");

 c2->cd(5);
 hist31->SetLineColor(kRed);
 plotHist(hist31,0.05,ncut1,ncut2);
 hist32->SetLineColor(kBlue);
 hist32->Draw("same");

 c2->cd(6);
 gPad->SetLogy();
 plotHist(hist31,0.05,ncut1,ncut2);
 hist32->Draw("same");

 c2->Print("trackletengine_barrel.pdf");

 c3->cd(1);
 hist41->SetLineColor(kRed);
 plotHist(hist41,0.05,ncut1,ncut2);
 hist42->SetLineColor(kBlue);
 hist42->Draw("same");

 c3->cd(2);
 gPad->SetLogy();
 plotHist(hist41,0.05,ncut1,ncut2);
 hist42->Draw("same");

 c3->cd(3);
 hist51->SetLineColor(kRed);
 plotHist(hist51,0.05,ncut1,ncut2);
 hist52->SetLineColor(kBlue);
 hist52->Draw("same");

 c3->cd(4);
 gPad->SetLogy();
 plotHist(hist51,0.05,ncut1,ncut2);
 hist52->Draw("same");

 c3->cd(5);
 hist61->SetLineColor(kRed);
 plotHist(hist61,0.05,ncut1,ncut2);
 hist62->SetLineColor(kBlue);
 hist62->Draw("same");

 c3->cd(6);
 gPad->SetLogy();
 plotHist(hist61,0.05,ncut1,ncut2);
 hist62->Draw("same");

 c3->Print("trackletengine_disk.pdf");



}
