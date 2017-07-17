#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include "MultiplicityPlotMacros.h"
#include "DPGAnalysis/SiStripTools/interface/CommonAnalyzer.h"
#include "TFile.h"
#include "TH1D.h"
#include "TProfile.h"
#include "TDirectory.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLegend.h"

void PlotPixelMultVtxPos(TFile* ff, const char* module) {

  CommonAnalyzer camult(ff,"",module);
  //  camult.setPath("VtxPosCorr");

  std::vector<std::string>  labels;
  labels.push_back("FPIX_m");
  labels.push_back("BPIX_L1_mod_1");
  labels.push_back("BPIX_L1_mod_2");
  labels.push_back("BPIX_L1_mod_3");
  labels.push_back("BPIX_L1_mod_4");
  labels.push_back("BPIX_L1_mod_5");
  labels.push_back("BPIX_L1_mod_6");
  labels.push_back("BPIX_L1_mod_7");
  labels.push_back("BPIX_L1_mod_8");
  labels.push_back("FPIX_p");
  labels.push_back("BPIX_L1");
  labels.push_back("BPIX_L2");
  labels.push_back("BPIX_L3");
  labels.push_back("Lumi");

  std::vector<TProfile*> profs;

  for(unsigned int i=0;i<labels.size();++i) {

    std::string path = "VtxPosCorr/"+labels[i];
    camult.setPath(path.c_str());

    std::string hname = "n"+labels[i]+"digivsvtxposprof";
    profs.push_back((TProfile*)camult.getObject(hname.c_str()));

  }

  TCanvas* cc = new TCanvas("BPIX L1 details","BPIX L1 details",1000,1000);
  gPad->Divide(2,2);
  
  for(unsigned int i = 1;i<5;++i) {
    cc->cd(i);
    if(profs[i] && profs[9-i]) {
      profs[i]->Draw(); 
      profs[9-i]->SetLineColor(kRed);
      profs[9-i]->SetMarkerColor(kRed);
      profs[9-i]->Draw("same"); 
      TLegend* leg = new TLegend(0.4,0.8,0.6,0.9,"Occupancy");
      leg->SetFillStyle(0);
      leg->AddEntry(profs[i],labels[i].c_str(),"l");
      leg->AddEntry(profs[9-i],labels[9-i].c_str(),"l");
      leg->Draw();
    }
  }
  new TCanvas("FPIX","FPIX");
    if(profs[0] && profs[9]) {
      profs[0]->Draw(); 
      profs[9]->SetLineColor(kRed);
      profs[9]->SetMarkerColor(kRed);
      profs[9]->Draw("same"); 
      TLegend* leg = new TLegend(0.4,0.8,0.6,0.9,"Occupancy");
      leg->SetFillStyle(0);
      leg->AddEntry(profs[0],labels[0].c_str(),"l");
      leg->AddEntry(profs[9],labels[9].c_str(),"l");
      leg->Draw();
    }

  gStyle->SetOptStat(11);
  gStyle->SetOptFit(11);
  new TCanvas("BPIXL1","BPIX L1");
  profs[10]->Fit("pol2");
  new TCanvas("BPIXL2","BPIX L2");
  profs[11]->Fit("pol2");
  new TCanvas("BPIXL3","BPIX L3");
  profs[12]->Fit("pol2");

  new TCanvas("LumiAdd","LumiAdd");
  TH1D* hlumi = profs[11]->ProjectionX("lumi");
  TH1D* hbpixl3 = profs[12]->ProjectionX("bpixl3");
  TH1D* hfpixm = profs[0]->ProjectionX("fpixm");
  TH1D* hfpixp = profs[9]->ProjectionX("fpixp");
  hlumi->SetTitle("BPIX L2+L3 + FPIX multiplicity vs vtx z position");
  hlumi->Add(hbpixl3);
  hlumi->Add(hfpixm);
  hlumi->Add(hfpixp);
  hlumi->Fit("pol2");
  new TCanvas("Lumi","Lumi");
  profs[13]->Fit("pol2");
}

TH1D* AverageRunMultiplicity(TFile& ff, const char* module, const bool excludeLastBins=false, const char* histo="nTKdigivsorbrun") {

  CommonAnalyzer camult(&ff,"",module);

  TH1D* clusmult = new TH1D("clusmult","Average Multiplicity vs run",10,0.,10.);
  clusmult->SetCanExtend(TH1::kXaxis);

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
	clusmult->Fill(runlabel,multvstime->GetMean(2));

      }
    }
  } 
  return clusmult;
}
