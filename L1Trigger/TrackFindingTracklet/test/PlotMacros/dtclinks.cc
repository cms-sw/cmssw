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
#include "TString.h"
#include "TText.h"
#include <string>
#include <map>


#include "plotHist.cc"


  

void dtclinks(){
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
  gStyle->SetLineWidth(2);
  gStyle->SetTextSize(1.1);
  gStyle->SetLabelSize(0.06,"xy");
  gStyle->SetTitleSize(0.06,"xy");
  gStyle->SetTitleOffset(1.2,"x");
  gStyle->SetTitleOffset(1.0,"y");
  gStyle->SetPadTopMargin(0.1);
  gStyle->SetPadRightMargin(0.1);
  gStyle->SetPadBottomMargin(0.16);
  gStyle->SetPadLeftMargin(0.12);



 TCanvas* c1 = new TCanvas("c1","Track performance",200,10,500,800);
 c1->Divide(1,2);
 c1->SetFillColor(0);
 c1->SetGrid();

 TCanvas* c2 = new TCanvas("c2","Track performance",200,10,700,800);
 c2->Divide(2,3);
 c2->SetFillColor(0);
 c2->SetGrid();

 TCanvas* c3 = new TCanvas("c3","Track performance",200,10,500,800);
 c3->Divide(1,3);
 c3->SetFillColor(0);
 c3->SetGrid();

 
 double max=200.0;
 
 TH1 *hist_all = new TH1F("All","All DTC Link Occupancy",max+1,-0.5,max+0.5);

 TH1 *hist_barrel = new TH1F("All","All DTC Link Occupancy (Barrel)",max+1,-0.5,max+0.5);
 TH1 *hist_disk = new TH1F("All","All DTC Link Occupancy (Disks)",max+1,-0.5,max+0.5);

 TH1 *hist_L1 = new TH1F("L1","L1 DTC Link Occupancy",max+1,-0.5,max+0.5);
 TH1 *hist_L2 = new TH1F("L2","L2 DTC Link Occupancy",max+1,-0.5,max+0.5);
 TH1 *hist_L3 = new TH1F("L3","L3 DTC Link Occupancy",max+1,-0.5,max+0.5);
 TH1 *hist_L4 = new TH1F("L4","L4 DTC Link Occupancy",max+1,-0.5,max+0.5);
 TH1 *hist_L5 = new TH1F("L5","L5 DTC Link Occupancy",max+1,-0.5,max+0.5);
 TH1 *hist_L6 = new TH1F("L6","L6 DTC Link Occupancy",max+1,-0.5,max+0.5);

 TH1 *hist_Ring13 = new TH1F("Ring 1-3","Ring 1-3 DTC Link Occupancy",max+1,-0.5,max+0.5);
 TH1 *hist_Ring49 = new TH1F("Ring 4-9","Ring 4-9 DTC Link Occupancy",max+1,-0.5,max+0.5);
 TH1 *hist_Ring1015 = new TH1F("Ring 10-15","Ring 10-15 DTC Link Occupancy",max+1,-0.5,max+0.5);


 ifstream in("dtclinks.txt");

 int count=0;

 std::map<string, TH1*> hists;

 while (in.good()){

   string dtc;
   int link;
   int stubs;
  
   in >>dtc>>link>>stubs;

   if (!in.good()) continue;

   if (stubs>max) stubs=max;

   hist_all->Fill(stubs);

   /*
   if (dtc<=143) hist_barrel->Fill(stubs);
   if (dtc>143) hist_disk->Fill(stubs);

   if (dtc>=0&&dtc<=15) hist_L1->Fill(stubs);
   if (dtc>=16&&dtc<=31) hist_L2->Fill(stubs);
   if (dtc>=32&&dtc<=63) hist_L3->Fill(stubs);
   if (dtc>=64&&dtc<=79) hist_L4->Fill(stubs);
   if (dtc>=80&&dtc<=111) hist_L5->Fill(stubs);
   if (dtc>=112&&dtc<=143) hist_L6->Fill(stubs);

   if (dtc>=144&&dtc<=159) hist_Ring13->Fill(stubs);   
   if (dtc>=160&&dtc<=191) hist_Ring49->Fill(stubs);   
   if (dtc>=192&&dtc<=255) hist_Ring1015->Fill(stubs);   
   */
   //cout << name <<" "<< countall <<" "<< countpass << endl;

   TString tindex=dtc+" ";
   tindex+=link;

   string index(tindex);
   
   std::map<string, TH1*>::iterator it=hists.find(index);

   if (it==hists.end()) {
     TString name="DTC = ";
     name+=dtc;
     name+=" Link = ";
     name+=link;
     TH1 *hist = new TH1F(name,name,max+1,-0.5,max+0.5);
     hist->Fill(stubs);
     hists[index]=hist;
   } else {
     hists[index]->Fill(stubs);
   }


   count++;

 }

 cout << "count = "<<count<<endl;

 c1->cd(1);
 plotHist(hist_all,0.03,83.0,50.0);
 c1->cd(2);
 plotHist(hist_disk,0.03,83.0,50.0);
 
 
 c1->Print("dtclinks.pdf(","pdf");

 c2->cd(1);
 plotHist(hist_L1,0.03,83.0,50.0);
 c2->cd(2);
 plotHist(hist_L2,0.03,83.0,50.0);
 c2->cd(3);
 plotHist(hist_L3,0.03,83.0,50.0);
 c2->cd(4);
 plotHist(hist_L4,0.03,83.0,50.0);
 c2->cd(5);
 plotHist(hist_L5,0.03,83.0,50.0);
 c2->cd(6);
 plotHist(hist_L6,0.03,83.0,50.0);

 c2->Print("dtclinks.pdf","pdf");

 c3->cd(1);
 plotHist(hist_Ring13,0.03,83.0,50.0);
 c3->cd(2);
 plotHist(hist_Ring49,0.03,83.0,50.0);
 c3->cd(3);
 plotHist(hist_Ring1015,0.03,83.0,50.0);

 c3->Print("dtclinks.pdf","pdf");
 
 int pages=0;

 std::map<string, TH1*>::iterator it=hists.begin();

 TCanvas* c=0;

 ofstream out("dtclinks.dat");

 bool closed=false;
 
 while(it!=hists.end()) {

   if (pages%4==0) {

     TString name = "DTC link : ";
     name+=it->first;
     
     c = new TCanvas(name,"Track performance",200,50,600,700);
     c->Divide(2,2);
     c->SetFillColor(0);
     c->SetGrid();

   }

   c->cd(pages%4+1);
   //sgPad->SetLogy();
   plotHist(it->second,0.03,83.0,50.0);

   if (it->second->GetMean()==0.0) {
     out << "Is empty : "<<it->first << endl;
   } else {
     out << "Not empty : "<<it->first << " "<<it->second->GetMean()<<endl;
   }
   
   
   pages++;

   ++it;

   if (pages%4==0) {
     if (it!=hists.end()) {
       c->Print("dtclinks.pdf","pdf");
     }
     else {
       closed=true;
       c->Print("dtclinks.pdf)","pdf");
     }
   }

 }

 if (!closed) {
   c->Print("dtclinks.pdf)","pdf");
 }
   
}
