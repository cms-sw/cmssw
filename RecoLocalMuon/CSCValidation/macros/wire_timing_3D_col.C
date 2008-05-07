{
#include <sstream>
#include <iostream>
#include <string.h>
 
gROOT->Reset();
//gROOT->ForceStyle();  //force the current style attributes to be set before
                      // reading an object from a file, see
                      // http://root.cern.ch/root/HowtoStyle.html
gROOT->SetStyle("Plain"); // to get rid of gray color of pad and have it white
//gStyle->SetOptTitle(0); // To get rid of histogram title
gStyle->SetPalette(1,0); // 
gStyle->SetOptStat(0);

std::ostringstream ss,ss1;

TFile f;
TFile f_in("validationHists_41356.root");
TFile f_out("wire_timing_41356.root","RECREATE");
std::string run="41356";

std::vector<std::string> xTitle;
xTitle.push_back("ME+1/1 CSC"); xTitle.push_back("ME+1/2 CSC");
xTitle.push_back("ME+1/3 CSC"); xTitle.push_back("ME+1/4 CSC");
xTitle.push_back("ME+2/1 CSC"); xTitle.push_back("ME+2/2 CSC");
xTitle.push_back("ME+3/1 CSC"); xTitle.push_back("ME+3/2 CSC");
xTitle.push_back("ME+4/1 CSC"); xTitle.push_back("ME+4/2 CSC");
xTitle.push_back("ME-1/1 CSC"); xTitle.push_back("ME-1/2 CSC");
xTitle.push_back("ME-1/3 CSC"); xTitle.push_back("ME-1/4 CSC");
xTitle.push_back("ME-2/1 CSC"); xTitle.push_back("ME-2/2 CSC");
xTitle.push_back("ME-3/1 CSC"); xTitle.push_back("ME-3/2 CSC");
xTitle.push_back("ME-4/1 CSC"); xTitle.push_back("ME-4/2 CSC");

std::vector<std::string> histName;
histName.push_back("_wire_timing_mean");
histName.push_back("_wire_timing_entries");

std::vector<std::string> histTitle;
histTitle.push_back("Mean Wire_Time_Bin");
histTitle.push_back("Entries Wire_Time_Bin");

std::string folder="GasGain/";

TH2F *h2[100];
TH2F *h;
Int_t esr[20]={111,112,113,114,121,122,131,132,141,142,
               211,212,213,214,221,222,231,232,241,242};

Int_t k=0;
TCanvas *c1=new TCanvas("c1","canvas");
c1->cd();
for(Int_t i=0;i<2;i++) 
for(Int_t j=0;j<20;j++) {
   ss<<folder.c_str()<<esr[j]<<histName[i].c_str();
   //   std::cout<<ss.str().c_str()<<std::endl;
   f_in.cd();
   TH2F *h2[k] = (TH2F*)f_in.Get(ss.str().c_str());
   ss.str("");
   if(h2[k] != NULL) {
     f_out.cd();
     ss<<"ME"<<esr[j]<<histName[i].c_str();
     ss1<<histTitle[i];
     TH2F *h=new TH2F(ss.str().c_str(),ss1.str().c_str(),40,0.0,40.0,42,1.0,43.0);
     ss.str("");
     ss1.str("");
     ss<<xTitle[j];
     h->GetXaxis()->SetTitle(ss.str().c_str());
     ss.str("");
     h->GetYaxis()->SetTitle("AFEB");
     h->GetZaxis()->SetLabelSize(0.03);
     h->SetOption("COLZ");
     Int_t nx=h2[k]->GetNbinsX();
     Int_t ny=h2[k]->GetNbinsY();
     for(Int_t in=1;in<=nx;in++) for(Int_t jn=1;jn<=ny;jn++) {
       Float_t w=h2[k]->GetBinContent(in,jn);
       h->SetBinContent(in,jn,w);
     }
     h->Draw();
     h->Write();
     c1->Update();     

     ss<<"ME"<<esr[j]<<histName[i].c_str()<<"_run_"<<run<<".gif";
     c1->Print(ss.str().c_str(),"gif");
     ss.str("");

     k++;
   }
}
}
