#include "DPGAnalysis/SiStripTools/bin/BSvsBPIX.h"
#include "TFile.h"
#include "TH1F.h"
#include "TProfile.h"
#include "TGraphErrors.h"
#include "TCanvas.h"
#include "TDirectory.h"
#include <iostream>

void BSvsBPIXPlot(TFile* ff, const char* bsmodule, const char* occumodule, const int run) {

  TGraphErrors* bspos = new TGraphErrors();
  TGraphErrors* bpixpos = new TGraphErrors();

  if(ff) {

    char bsfolder[200];
    sprintf(bsfolder,"%s/run_%d",bsmodule,run);
    if(ff->cd(bsfolder)) {
      TH1F* bsx = (TH1F*)gDirectory->Get("bsxrun");
      TH1F* bsy = (TH1F*)gDirectory->Get("bsyrun");
      if(bsx && bsy) {
	std::cout << "beam spot position (" 
		  << bsx->GetMean() << "+/-" << bsx->GetMeanError() << ","
		  << bsy->GetMean() << "+/-" << bsy->GetMeanError() << ")" << std::endl;
	bspos->SetPoint(0,bsx->GetMean(),bsy->GetMean());
	bspos->SetPointError(0,bsx->GetMeanError(),bsy->GetMeanError());
      }
    }
    char occufolder[200];
    sprintf(occufolder,"%s/run_%d",occumodule,run);
    if(ff->cd(occufolder)) {
      TProfile* xmean = (TProfile*)gDirectory->Get("avex");
      TProfile* ymean = (TProfile*)gDirectory->Get("avey");
      if(xmean && ymean) {
	for(int i=1;i<=xmean->GetNbinsX();++i) {
	  if(xmean->GetBinEntries(i) >0) {
	    std::cout << "ladder position " << i << " : ("
		      << xmean->GetBinContent(i) << "+/-" << xmean->GetBinError(i) << ","
		      << ymean->GetBinContent(i) << "+/-" << ymean->GetBinError(i) << ")" << std::endl;
	    int point = bpixpos->GetN();
	    bpixpos->SetPoint(point,xmean->GetBinContent(i),ymean->GetBinContent(i));
	    bpixpos->SetPointError(point,xmean->GetBinError(i),ymean->GetBinError(i));
	  }
	}
      }
      
    }
  }
  new TCanvas("bsbpix","bsbpix",500,500);
  bpixpos->Draw("ap");
  bspos->Draw("p");

}
