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



void vmoccupancy(TString subname="me"){

//
// To see the output of this macro, click here.

//

#include "TMath.h"

  if (!(subname=="me"||subname=="te")) {
    cout << "Argument to vmoccupancy should be either 'me' or 'te'"<<endl; 
      return;  
  }

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




 TCanvas* c1 = new TCanvas("c1","Track performance",200,10,700,800);
 c1->Divide(2,3);
 c1->SetFillColor(0);
 c1->SetGrid();

 TCanvas* c2 = new TCanvas("c2","Track performance",200,10,700,800);
 c2->Divide(2,3);
 c2->SetFillColor(0);
 c2->SetGrid();

 TCanvas* c3 = new TCanvas("c3","Track performance",200,10,700,800);
 c3->Divide(3,3);
 c3->SetFillColor(0);
 c3->SetGrid();

 
 TH1 *hist1 = new TH1F("h1","Stubs per VM L1",30,0.0,30.0);
 TH1 *hist2 = new TH1F("h2","Stubs per VM L2",30,0.0,30.0);
 TH1 *hist3 = new TH1F("h3","Stubs per VM L3",30,0.0,30.0);
 TH1 *hist4 = new TH1F("h4","Stubs per VM L4",30,0.0,30.0);
 TH1 *hist5 = new TH1F("h5","Stubs per VM L5",30,0.0,30.0);
 TH1 *hist6 = new TH1F("h6","Stubs per VM L6",30,0.0,30.0);

 TH1 *hist11 = new TH1F("h11","Stubs per VM D1",30,0.0,30.0);
 TH1 *hist12 = new TH1F("h12","Stubs per VM D2",30,0.0,30.0);
 TH1 *hist13 = new TH1F("h13","Stubs per VM D3",30,0.0,30.0);
 TH1 *hist14 = new TH1F("h14","Stubs per VM D4",30,0.0,30.0);
 TH1 *hist15 = new TH1F("h15","Stubs per VM D5",30,0.0,30.0);

 TH1 *hist21 = new TH1F("h21","Disk bin 1",30,0.0,30.0);
 TH1 *hist22 = new TH1F("h22","Disk bin 2",30,0.0,30.0);
 TH1 *hist23 = new TH1F("h23","Disk bin 3",30,0.0,30.0);
 TH1 *hist24 = new TH1F("h24","Disk bin 4",30,0.0,30.0);
 TH1 *hist25 = new TH1F("h25","Disk bin 5",30,0.0,30.0);
 TH1 *hist26 = new TH1F("h26","Disk bin 6",30,0.0,30.0);
 TH1 *hist27 = new TH1F("h27","Disk bin 7",30,0.0,30.0);
 TH1 *hist28 = new TH1F("h28","Disk bin 8",30,0.0,30.0);
 

 ifstream in("vmoccupancy"+subname+".txt");

 int count=0;

 while (in.good()){
   
   int nstub, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16;
   string name;
   
   in >>name>>nstub>>n1>>n2>>n3>>n4>>n5>>n6>>n7>>n8>>n9>>n10>>n11>>n12>>n13>>n14>>n15>>n16;
   
   //cout <<layer<<" "<<r<<endl;

   if (name[6]=='L') {

     int layer=name[7]-'0';

     if (layer==1) hist1->Fill(nstub);
     if (layer==2) hist2->Fill(nstub);
     if (layer==3) hist3->Fill(nstub);
     if (layer==4) hist4->Fill(nstub);
     if (layer==5) hist5->Fill(nstub);
     if (layer==6) hist6->Fill(nstub);
   }

   if (name[6]=='D') {

     int disk=name[7]-'0';

     if (disk==1) hist11->Fill(nstub);
     if (disk==2) hist12->Fill(nstub);
     if (disk==3) hist13->Fill(nstub);
     if (disk==4) hist14->Fill(nstub);
     if (disk==5) hist15->Fill(nstub);

     if (disk==2) {
       hist21->Fill(n1);
       hist21->Fill(n9);
       
       hist22->Fill(n2);
       hist22->Fill(n10);

       hist23->Fill(n3);
       hist23->Fill(n11);

       hist24->Fill(n4);
       hist24->Fill(n12);

       hist25->Fill(n5);
       hist25->Fill(n13);
       
       hist26->Fill(n6);
       hist26->Fill(n14);

       hist27->Fill(n7);
       hist27->Fill(n15);

       hist28->Fill(n8);
       hist28->Fill(n16);
     }
       
   }



   count++;

 }

 cout << "Processed: "<<count<<" events"<<endl;

 bool logy=false;

 c1->cd(1);
 if (logy) gPad->SetLogy();
 TF1* f1 = new TF1("f1","[0]*TMath::Power([1],x)*(TMath::Exp(-[1]))/TMath::Gamma(x+1)", 0, 30);
 f1->SetParameters(1.0, 1.0); 
 //hist1->Fit("f1", "R"); 
 hist1->Draw();

 c1->cd(2);
 if (logy)  gPad->SetLogy();
 TF1* f2 = new TF1("f2","[0]*TMath::Power([1],x)*(TMath::Exp(-[1]))/TMath::Gamma(x+1)", 0, 30);
 f2->SetParameters(1.0, 1.0); 
 //hist2->Fit("f2", "R"); 
 hist2->Draw();

 c1->cd(3);
 if (logy)  gPad->SetLogy();
 TF1* f3 = new TF1("f3","[0]*TMath::Power([1],x)*(TMath::Exp(-[1]))/TMath::Gamma(x+1)", 0, 30);
 f3->SetParameters(1.0, 1.0); 
 //hist3->Fit("f3", "R"); 
 hist3->Draw();

 c1->cd(4);
 if (logy)  gPad->SetLogy();
 TF1* f4 = new TF1("f4","[0]*TMath::Power([1],x)*(TMath::Exp(-[1]))/TMath::Gamma(x+1)", 0, 30);
 f4->SetParameters(1.0, 1.0); 
 //hist4->Fit("f4", "R"); 
 hist4->Draw();

 c1->cd(5);
 if (logy)  gPad->SetLogy();
 TF1* f5 = new TF1("f5","[0]*TMath::Power([1],x)*(TMath::Exp(-[1]))/TMath::Gamma(x+1)", 0, 30);
 f5->SetParameters(1.0, 1.0); 
 //hist5->Fit("f5", "R"); 
 hist5->Draw();

 c1->cd(6);
 if (logy)  gPad->SetLogy();
 TF1* f6 = new TF1("f6","[0]*TMath::Power([1],x)*(TMath::Exp(-[1]))/TMath::Gamma(x+1)", 0, 30);
 f6->SetParameters(1.0, 1.0); 
 //hist6->Fit("f6", "R"); 
 hist6->Draw();


 c1->Print("vmoccupancy"+subname+".pdf(","pdf");

 c2->cd(1);
 if (logy) gPad->SetLogy();
 TF1* f11 = new TF1("f11","[0]*TMath::Power([1],x)*(TMath::Exp(-[1]))/TMath::Gamma(x+1)", 0, 30);
 f11->SetParameters(1.0, 1.0); 
 //hist1->Fit("f1", "R"); 
 hist11->Draw();

 c2->cd(2);
 if (logy)  gPad->SetLogy();
 TF1* f12 = new TF1("f12","[0]*TMath::Power([1],x)*(TMath::Exp(-[1]))/TMath::Gamma(x+1)", 0, 30);
 f12->SetParameters(1.0, 1.0); 
 //hist2->Fit("f2", "R"); 
 hist12->Draw();

 c2->cd(3);
 if (logy)  gPad->SetLogy();
 TF1* f13 = new TF1("f13","[0]*TMath::Power([1],x)*(TMath::Exp(-[1]))/TMath::Gamma(x+1)", 0, 30);
 f13->SetParameters(1.0, 1.0); 
 //hist3->Fit("f3", "R"); 
 hist13->Draw();

 c2->cd(4);
 if (logy)  gPad->SetLogy();
 TF1* f14 = new TF1("f14","[0]*TMath::Power([1],x)*(TMath::Exp(-[1]))/TMath::Gamma(x+1)", 0, 30);
 f14->SetParameters(1.0, 1.0); 
 //hist4->Fit("f4", "R"); 
 hist14->Draw();

 c2->cd(5);
 if (logy)  gPad->SetLogy();
 TF1* f15 = new TF1("f15","[0]*TMath::Power([1],x)*(TMath::Exp(-[1]))/TMath::Gamma(x+1)", 0, 30);
 f15->SetParameters(1.0, 1.0); 
 //hist5->Fit("f5", "R"); 
 hist15->Draw();

 c2->Print("vmoccupancy"+subname+".pdf)","pdf");

 c3->cd(1);
 gPad->SetLogy();
 hist21->Draw();

 c3->cd(2);
 gPad->SetLogy();
 hist22->Draw();

 c3->cd(3);
 gPad->SetLogy();
 hist23->Draw();

 c3->cd(4);
 gPad->SetLogy();
 hist24->Draw();

 c3->cd(5);
 gPad->SetLogy();
 hist25->Draw();

 c3->cd(6);
 gPad->SetLogy();
 hist26->Draw();

 c3->cd(7);
 gPad->SetLogy();
 hist27->Draw();

 c3->cd(8);
 gPad->SetLogy();
 hist28->Draw();

 c3->Print("vmoccupancydiskbins"+subname+".pdf");

 
}

