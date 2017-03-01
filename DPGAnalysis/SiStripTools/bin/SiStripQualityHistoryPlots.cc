#include "SiStripQualityHistoryPlots.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <map>
#include "TPad.h"
#include "TFile.h"
#include "TH2F.h"
#include "TH1F.h"
#include "TProfile.h"
#include "TGraph.h"
#include "DPGAnalysis/SiStripTools/interface/CommonAnalyzer.h"
#include "TCanvas.h"
#include "TStyle.h"


TH1D* AverageRunBadChannels(TFile& ff, const char* module, const char* histo, const bool excludeLastBins) {

  CommonAnalyzer camult(&ff,"",module);

  TH1D* badchannels = new TH1D("badchannels","Average Number of Bad Channels vs run",10,0.,10.);
  badchannels->SetCanExtend(TH1::kXaxis);

  std::vector<unsigned int> runs = camult.getRunList();
  std::sort(runs.begin(),runs.end());
  
  {
    for(unsigned int i=0;i<runs.size();++i) {
      
      char runlabel[100];
      sprintf(runlabel,"%d",runs[i]);
      char runpath[100];
      sprintf(runpath,"run_%d",runs[i]);
      camult.setPath(runpath);
      
      
      TProfile* multvstime=0;
      if(multvstime==0) multvstime = (TProfile*)camult.getObject(histo);
      if(multvstime) {
	// compute mean exlucing the last filled bins

	if(excludeLastBins) {
	  int lastbin= multvstime->GetNbinsX()+1;
	  int firstbin= 1;
	  for(int ibin=multvstime->GetNbinsX()+1;ibin>0;--ibin) {
	    if(multvstime->GetBinEntries(ibin)!=0) {
	      lastbin=ibin;
	      break;
	    }
	  }
	  
	  std::cout << "Restricted range: " << firstbin << " " << lastbin << std::endl;
	  multvstime->GetXaxis()->SetRangeUser(multvstime->GetBinLowEdge(firstbin),multvstime->GetBinLowEdge(lastbin-2));
	}
	// fill the summary
	badchannels->Fill(runlabel,multvstime->GetMean(2));

      }
    }
  } 
  return badchannels;
  


}

TCanvas* StripCompletePlot(TFile& ff, const char* module, const bool excludeLastBins) {

  TCanvas* cc = new TCanvas();

  TH1D* cabling = AverageRunBadChannels(ff,module,"badmodrun_Cabling",excludeLastBins);
  TH1D* runinfo = AverageRunBadChannels(ff,module,"badmodrun_RunInfo",excludeLastBins);
  TH1D* badchannel = AverageRunBadChannels(ff,module,"badmodrun_BadChannel",excludeLastBins);
  TH1D* dcs = AverageRunBadChannels(ff,module,"badmodrun_DCS",excludeLastBins);
  TH1D* badfiber = AverageRunBadChannels(ff,module,"badmodrun_BadFiber",excludeLastBins);

  cabling->SetLineColor(kRed);
  runinfo->SetLineColor(kMagenta);
  badchannel->SetLineColor(kCyan);
  dcs->SetLineColor(kGreen);
  badfiber->SetLineColor(kBlue);

  badfiber->Draw();
  dcs->Draw("same");
  badchannel->Draw("same");
  runinfo->Draw("same");
  cabling->Draw("same");

  return cc;
}
