#include <TFile.h>
#include <TH1D.h>
#include <TH1I.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TPaveStats.h>
#include <TPaveText.h>
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cstdlib>

void PlotCollapse(const char* infile, std::string text, int tag=311,
		  bool drawStatBox=true, bool save=false) {
  std::string name[6] = {"h_merge","h_size","h_depth","h_sfrac","h_frac",
			 "h_balance"};
  int         type[6] = {0,0,0,0,1,1};       
  std::string xtitl[6]= {"Merged Hits", "Size of RecHit Collections",
			 "Depth in DetId", 
			 "Ratio of collection sizes #frac{RecHit}{PreRecHit}",
			 "Energy fraction of collapsed hits",
			 "Energy balance between pre- and post-collapse"};
  std::string ytitl[6]= {"Hits","Events","Hits","Events","Hits","Hits"};

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  if (drawStatBox) {
    gStyle->SetOptStat(111110);  gStyle->SetOptFit(1);
  } else {
    gStyle->SetOptStat(0);       gStyle->SetOptFit(0);
  }

  TFile      *file = new TFile(infile);
  if (file) {
    for (unsigned int k=0; k<6; ++k) {
      TH1D* hist = (TH1D*)file->FindObjectAny(name[k].c_str());
      if (hist) {
	char namep[100];
	sprintf (namep, "c%s%d", name[k].c_str(), tag);
	TCanvas *pad = new TCanvas(namep, namep, 700, 500);
	pad->SetRightMargin(0.10);
	pad->SetTopMargin(0.10);
	if (type[k] == 1) pad->SetLogy();
	hist->GetXaxis()->SetTitleSize(0.04);
	hist->GetXaxis()->SetTitle(xtitl[k].c_str());
	hist->GetYaxis()->SetTitle(ytitl[k].c_str());
	hist->GetYaxis()->SetLabelOffset(0.005);
	hist->GetYaxis()->SetTitleSize(0.04);
	hist->GetYaxis()->SetLabelSize(0.035);
	hist->GetYaxis()->SetTitleOffset(1.10);
	hist->SetMarkerStyle(20);
	hist->SetMarkerColor(2);
	hist->SetLineColor(2);
	hist->Draw();
	pad->Update();
	TPaveStats* st1 = (TPaveStats*)hist->GetListOfFunctions()->FindObject("stats");
	if (st1 != NULL) {
	  st1->SetY1NDC(0.70); st1->SetY2NDC(0.90);
	  st1->SetX1NDC(0.65); st1->SetX2NDC(0.90);
	}
	TPaveText *txt1 = new TPaveText(0.25,0.91,0.90,0.96,"blNDC");
	txt1->SetFillColor(0);
	txt1->AddText(text.c_str());
	txt1->Draw("same");
	pad->Modified();
	pad->Update();
	if (save) {
	  sprintf (namep, "%s.pdf", pad->GetName());
	  pad->Print(namep);
	}
      }
    }
  }
}
