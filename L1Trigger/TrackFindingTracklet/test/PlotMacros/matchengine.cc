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
#include "plotHist.cc"
#include <string>
#include <map>



void matchengine(){


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
  c1->Divide(2,3);
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
  
  TCanvas* c4 = new TCanvas("c4","Track performance",200,10,700,800);
  c4->Divide(2,3);
  c4->SetFillColor(0);
  c4->SetGrid();
  
  
  double max=128.0;
  
  TH1 *hist_L1 = new TH1F("L1","Matches per VM",max+1,-0.5,max+0.5);
  TH1 *hist_L2 = new TH1F("L2","Matches per VM",max+1,-0.5,max+0.5);
  TH1 *hist_L3 = new TH1F("L3","Matches per VM",max+1,-0.5,max+0.5);
  TH1 *hist_L4 = new TH1F("L4","Matches per VM",max+1,-0.5,max+0.5);
  TH1 *hist_L5 = new TH1F("L5","Matches per VM",max+1,-0.5,max+0.5);
  TH1 *hist_L6 = new TH1F("L6","Matches per VM",max+1,-0.5,max+0.5);

  TH1 *hist_D1 = new TH1F("D1","Matches per VM",max+1,-0.5,max+0.5);
  TH1 *hist_D2 = new TH1F("D2","Matches per VM",max+1,-0.5,max+0.5);
  TH1 *hist_D3 = new TH1F("D3","Matches per VM",max+1,-0.5,max+0.5);
  TH1 *hist_D4 = new TH1F("D4","Matches per VM",max+1,-0.5,max+0.5);
  TH1 *hist_D5 = new TH1F("D5","Matches per VM",max+1,-0.5,max+0.5);

  TH1 *hist_L1region = new TH1F("L1 Region","Matches per region",max+1,-0.5,4*max+0.5);
  TH1 *hist_L2region = new TH1F("L2 Region","Matches per region",max+1,-0.5,8*max+0.5);
  TH1 *hist_L3region = new TH1F("L3 Region","Matches per region",max+1,-0.5,8*max+0.5);
  TH1 *hist_L4region = new TH1F("L4 Region","Matches per region",max+1,-0.5,8*max+0.5);
  TH1 *hist_L5region = new TH1F("L5 Region","Matches per region",max+1,-0.5,8*max+0.5);
  TH1 *hist_L6region = new TH1F("L6 Region","Matches per region",max+1,-0.5,8*max+0.5);

  TH1 *hist_D1region = new TH1F("D1 Region","Matches per region",max+1,-0.5,8*max+0.5);
  TH1 *hist_D2region = new TH1F("D2 Region","Matches per region",max+1,-0.5,4*max+0.5);
  TH1 *hist_D3region = new TH1F("D3 Region","Matches per region",max+1,-0.5,4*max+0.5);
  TH1 *hist_D4region = new TH1F("D4 Region","Matches per region",max+1,-0.5,4*max+0.5);
  TH1 *hist_D5region = new TH1F("D5 Region","Matches per region",max+1,-0.5,4*max+0.5); 

  TH1 *hist_L1regionpass = new TH1F("L1 Region Max","Matches per region",max+1,-0.5,4*max+0.5);
  TH1 *hist_L2regionpass = new TH1F("L2 Region Max","Matches per region",max+1,-0.5,8*max+0.5);
  TH1 *hist_L3regionpass = new TH1F("L3 Region Max","Matches per region",max+1,-0.5,8*max+0.5);
  TH1 *hist_L4regionpass = new TH1F("L4 Region Max","Matches per region",max+1,-0.5,8*max+0.5);
  TH1 *hist_L5regionpass = new TH1F("L5 Region Max","Matches per region",max+1,-0.5,8*max+0.5);
  TH1 *hist_L6regionpass = new TH1F("L6 Region Max","Matches per region",max+1,-0.5,8*max+0.5);

  TH1 *hist_D1regionpass = new TH1F("D1 Region Max","Matches per region",max+1,-0.5,8*max+0.5);
  TH1 *hist_D2regionpass = new TH1F("D2 Region Max","Matches per region",max+1,-0.5,4*max+0.5);
  TH1 *hist_D3regionpass = new TH1F("D3 Region Max","Matches per region",max+1,-0.5,4*max+0.5);
  TH1 *hist_D4regionpass = new TH1F("D4 Region Max","Matches per region",max+1,-0.5,4*max+0.5);
  TH1 *hist_D5regionpass = new TH1F("D5 Region Max","Matches per region",max+1,-0.5,4*max+0.5);

  
  TH1 *hist_L1pass = new TH1F("L1","Matches per VM pass",max+1,-0.5,max+0.5);
  TH1 *hist_L2pass = new TH1F("L2","Matches per VM pass",max+1,-0.5,max+0.5);
  TH1 *hist_L3pass = new TH1F("L3","Matches per VM pass",max+1,-0.5,max+0.5);
  TH1 *hist_L4pass = new TH1F("L4","Matches per VM pass",max+1,-0.5,max+0.5);
  TH1 *hist_L5pass = new TH1F("L5","Matches per VM pass",max+1,-0.5,max+0.5);
  TH1 *hist_L6pass = new TH1F("L6","Matches per VM pass",max+1,-0.5,max+0.5);

  TH1 *hist_D1pass = new TH1F("D1","Matches per VM pass",max+1,-0.5,max+0.5);
  TH1 *hist_D2pass = new TH1F("D2","Matches per VM pass",max+1,-0.5,max+0.5);
  TH1 *hist_D3pass = new TH1F("D3","Matches per VM pass",max+1,-0.5,max+0.5);
  TH1 *hist_D4pass = new TH1F("D4","Matches per VM pass",max+1,-0.5,max+0.5);
  TH1 *hist_D5pass = new TH1F("D5","Matches per VM pass",max+1,-0.5,max+0.5);

  ifstream in("matchengine.txt");

  int count=0;

  std::map<TString, TH1*> hists;

  TString nameold="";
  int regioncount=0;
  int regioncountpass=0;
  
  while (in.good()){

    TString name;
    int matchesall;
    int matchespass;
    
    in >>name>>matchesall>>matchespass;

    TSubString sub=name(0,9);
    TString subname(sub);
    cout << "subname "<<subname<<endl;

    if (nameold=="") {
      nameold=subname;
    }

    if (nameold==subname) {
      regioncount+=matchesall;      
      regioncountpass+=matchespass;      
    } else {
      if (name.Contains("L1PHI")) {
	if (regioncount>4*max) regioncount=4*max;
	if (regioncountpass>4*max) regioncountpass=4*max;
	hist_L1region->Fill(regioncount);
	hist_L1regionpass->Fill(regioncountpass);
      }
      if (name.Contains("L2PHI")) {
	if (regioncount>8*max) regioncount=8*max;
	if (regioncountpass>8*max) regioncountpass=8*max;
	hist_L2region->Fill(regioncount);
	hist_L2regionpass->Fill(regioncountpass);
      }
      if (name.Contains("L3PHI")) {
	if (regioncount>8*max) regioncount=8*max;
	if (regioncountpass>8*max) regioncountpass=8*max;
	hist_L3region->Fill(regioncount);
	hist_L3regionpass->Fill(regioncountpass);
      }
      if (name.Contains("L4PHI")) {
	if (regioncount>8*max) regioncount=8*max;
	if (regioncountpass>8*max) regioncountpass=8*max;
	hist_L4region->Fill(regioncount);
	hist_L4regionpass->Fill(regioncountpass);
      }
      if (name.Contains("L5PHI")) {
	if (regioncount>8*max) regioncount=8*max;
	if (regioncountpass>8*max) regioncountpass=8*max;
	hist_L5region->Fill(regioncount);
	hist_L5regionpass->Fill(regioncountpass);
      }
      if (name.Contains("L6PHI")) {
	if (regioncount>8*max) regioncount=8*max;
	if (regioncountpass>8*max) regioncountpass=8*max;
	hist_L6region->Fill(regioncount);
	hist_L6regionpass->Fill(regioncountpass);
      }

      if (name.Contains("D1PHI")) {
	if (regioncount>8*max) regioncount=8*max;
	if (regioncountpass>8*max) regioncountpass=8*max;
	hist_D1region->Fill(regioncount);
	hist_D1regionpass->Fill(regioncountpass);
      }
      if (name.Contains("D2PHI")) {
	if (regioncount>4*max) regioncount=4*max;
	if (regioncountpass>4*max) regioncountpass=4*max;
	hist_D2region->Fill(regioncount);
	hist_D2regionpass->Fill(regioncountpass);
      }
      if (name.Contains("D3PHI")) {
	if (regioncount>4*max) regioncount=4*max;
	if (regioncountpass>4*max) regioncountpass=4*max;
	hist_D3region->Fill(regioncount);
	hist_D3regionpass->Fill(regioncountpass);
      }
      if (name.Contains("D4PHI")) {
	if (regioncount>4*max) regioncount=4*max;
	if (regioncountpass>4*max) regioncountpass=4*max;
	hist_D4region->Fill(regioncount);
	hist_D4regionpass->Fill(regioncountpass);
      }
      if (name.Contains("D5PHI")) {
	if (regioncount>4*max) regioncount=4*max;
	if (regioncountpass>4*max) regioncountpass=4*max;
	hist_D5region->Fill(regioncount);
	hist_D5regionpass->Fill(regioncountpass);
      }
      
      nameold=subname;
      regioncount=matchesall;      
      regioncountpass=matchespass;      
    }
    
    
    if (!in.good()) continue;
    
    if (matchesall>max) matchesall=max;
    if (matchespass>max) matchespass=max;
    
    if (name.Contains("L1PHI")) {
      hist_L1->Fill(matchesall);
      hist_L1pass->Fill(matchespass);
    }
    if (name.Contains("L2PHI")) {
      hist_L2->Fill(matchesall);
      hist_L2pass->Fill(matchespass);
    }
    if (name.Contains("L3PHI")) {
      hist_L3->Fill(matchesall);
      hist_L3pass->Fill(matchespass);
    }
    if (name.Contains("L4PHI")) {
      hist_L4->Fill(matchesall);
      hist_L4pass->Fill(matchespass);
    }
    if (name.Contains("L5PHI")) {
      hist_L5->Fill(matchesall);
      hist_L5pass->Fill(matchespass);
    }
    if (name.Contains("L6PHI")) {
      hist_L6->Fill(matchesall);
      hist_L6pass->Fill(matchespass);
    }
    
    if (name.Contains("D1PHI")) {
      hist_D1->Fill(matchesall);
      hist_D1pass->Fill(matchespass);
    }
    if (name.Contains("D2PHI")) {
      hist_D2->Fill(matchesall);
      hist_D2pass->Fill(matchespass);
    }
    if (name.Contains("D3PHI")) {
      hist_D3->Fill(matchesall);
      hist_D3pass->Fill(matchespass);
    }
    if (name.Contains("D4PHI")) {
      hist_D4->Fill(matchesall);
      hist_D4pass->Fill(matchespass);
    }
    if (name.Contains("D5PHI")) {
      hist_D5->Fill(matchesall);
      hist_D5pass->Fill(matchespass);
    }


    std::map<TString, TH1*>::iterator it=hists.find(name);

    if (it==hists.end()) {
      TH1 *hist = new TH1F(name,name,max+1,-0.5,max+0.5);
      hist->Fill(matchesall);
      hists[name]=hist;
      cout << "hists size = "<<hists.size()<<" "<<name<<endl;
    } else {
      hists[name]->Fill(matchesall);
    }

    count++;

 }

  cout << "count = "<<count<<endl;


  c1->cd(1);
  gPad->SetLogy();
  plotHist(hist_L1,0.05,ncut1,ncut2);
  hist_L1pass->SetLineColor(kBlue);
  hist_L1pass->Draw("same");
  
  c1->cd(2);
  gPad->SetLogy();
  plotHist(hist_L2,0.05,ncut1,ncut2);
  hist_L2pass->SetLineColor(kBlue);
  hist_L2pass->Draw("same");


  c1->cd(3);
  gPad->SetLogy();
  plotHist(hist_L3,0.05,ncut1,ncut2);
  hist_L3pass->SetLineColor(kBlue);
  hist_L3pass->Draw("same");


  c1->cd(4);
  gPad->SetLogy();
  plotHist(hist_L4,0.05,ncut1,ncut2);
  hist_L4pass->SetLineColor(kBlue);
  hist_L4pass->Draw("same");


  c1->cd(5);
  gPad->SetLogy();
  plotHist(hist_L5,0.05,ncut1,ncut2);
  hist_L5pass->SetLineColor(kBlue);
  hist_L5pass->Draw("same");


  c1->cd(6);
  gPad->SetLogy();
  plotHist(hist_L6,0.05,ncut1,ncut2);
  hist_L6pass->SetLineColor(kBlue);
  hist_L6pass->Draw("same");


  c1->Print("matchengine.pdf(","pdf");

  c2->cd(1);
  gPad->SetLogy();
  plotHist(hist_D1,0.05,ncut1,ncut2);
  hist_D1pass->SetLineColor(kBlue);
  hist_D1pass->Draw("same");

  
  c2->cd(2);
  gPad->SetLogy();
  plotHist(hist_D2,0.05,ncut1,ncut2);
  hist_D2pass->SetLineColor(kBlue);
  hist_D2pass->Draw("same");

  c2->cd(3);
  gPad->SetLogy();
  plotHist(hist_D3,0.05,ncut1,ncut2);
  hist_D3pass->SetLineColor(kBlue);
  hist_D3pass->Draw("same");
  
  c2->cd(4);
  gPad->SetLogy();
  plotHist(hist_D4,0.05,ncut1,ncut2);
  hist_D4pass->SetLineColor(kBlue);
  hist_D4pass->Draw("same");

  
  c2->cd(5);
  gPad->SetLogy();
  plotHist(hist_D5,0.05,ncut1,ncut2);
  hist_D5pass->SetLineColor(kBlue);
  hist_D5pass->Draw("same");

  
  c2->Print("matchengine.pdf","pdf");
  
  c3->cd(1);
  gPad->SetLogy();
  plotHist(hist_L1region,0.05,ncut1,ncut2);
  hist_L1regionpass->SetLineColor(kBlue);
  hist_L1regionpass->Draw("same");

  c3->cd(2);
  gPad->SetLogy();
  plotHist(hist_L2region,0.05,ncut1,ncut2);
  hist_L2regionpass->SetLineColor(kBlue);
  hist_L2regionpass->Draw("same");

  c3->cd(3);
  gPad->SetLogy();
  plotHist(hist_L3region,0.05,ncut1,ncut2);
  hist_L3regionpass->SetLineColor(kBlue);
  hist_L3regionpass->Draw("same");

  c3->cd(4);
  gPad->SetLogy();
  plotHist(hist_L4region,0.05,ncut1,ncut2);
  hist_L4regionpass->SetLineColor(kBlue);
  hist_L4regionpass->Draw("same");

  c3->cd(5);
  gPad->SetLogy();
  plotHist(hist_L5region,0.05,ncut1,ncut2);
  hist_L5regionpass->SetLineColor(kBlue);
  hist_L5regionpass->Draw("same");

  c3->cd(6);
  gPad->SetLogy();
  plotHist(hist_L6region,0.05,ncut1,ncut2);
  hist_L6regionpass->SetLineColor(kBlue);
  hist_L6regionpass->Draw("same");

  
  c3->Print("matchengine.pdf","pdf");
  
  c4->cd(1);
  gPad->SetLogy();
  plotHist(hist_D1region,0.05,ncut1,ncut2);
  hist_D1regionpass->SetLineColor(kBlue);
  hist_D1regionpass->Draw("same");

  c4->cd(2);
  gPad->SetLogy();
  plotHist(hist_D2region,0.05,ncut1,ncut2);
  hist_D2regionpass->SetLineColor(kBlue);
  hist_D2regionpass->Draw("same");

  c4->cd(3);
  gPad->SetLogy();
  plotHist(hist_D3region,0.05,ncut1,ncut2);
  hist_D3regionpass->SetLineColor(kBlue);
  hist_D3regionpass->Draw("same");

  c4->cd(4);
  gPad->SetLogy();
  plotHist(hist_D4region,0.05,ncut1,ncut2);
  hist_D4regionpass->SetLineColor(kBlue);
  hist_D4regionpass->Draw("same");

  c4->cd(5);
  gPad->SetLogy();
  plotHist(hist_D5region,0.05,ncut1,ncut2);
  hist_D5regionpass->SetLineColor(kBlue);
  hist_D5regionpass->Draw("same");

   
  c4->Print("matchengine.pdf","pdf");

  int pages=0;

  std::map<TString, TH1*>::iterator it=hists.begin();

  TCanvas* c=0;
  
  
  while(it!=hists.end()) {
    
    if (pages%4==0) {
      
     c = new TCanvas(it->first,"Track performance",200,50,600,700);
     c->Divide(2,2);
     c->SetFillColor(0);
     c->SetGrid();

   }

   c->cd(pages%4+1);
   //gPad->SetLogy();
   plotHist(it->second,0.05,ncut1,ncut2);

   pages++;

   ++it;

   if (it==hists.end()) {
     c->Print("matchengine.pdf)","pdf");
   }
   else {
     if (pages%4==0) {
       c->Print("matchengine.pdf","pdf");
     }
   }

   


 }

  //c->Print("matchengine.pdf)","pdf");

}
