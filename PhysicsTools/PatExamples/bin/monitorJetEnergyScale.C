#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "TF1.h"
#include "TH1F.h"
#include "TFile.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TLegend.h"

/// maximal number of bins used for the jet
/// response plots
static const unsigned int MAXBIN=8;
/// binning used for the jet response plots 
/// (NOTE BINS must have a length of MAXBIN
/// +1)
static const float BINS[]={30., 40., 50., 60., 70., 80., 100., 125., 150.};

/// -------------------------------------------------------------------------------
///
/// Determine and display the Jet Response for Uncorrected, L2Relative and 
/// L3Absolute corrected jets as a function of the pt of the matched genJet
/// from a set of basic histograms filled in the PatJetAnalyzer. The mean 
/// jet energy response is determined from simple gaussien fits w/o any 
/// extras.
/// The use case is:
/// .x PhysicsTools/PatExamples/bin/monitorJetEnergyScale.C+
///
/// -------------------------------------------------------------------------------
void monitorJetEnergyScale()
{
  gStyle->SetOptStat(   0);
  gStyle->SetOptFit (1111);

  // list of valid histogram names
  std::vector<std::string> corrLevels_;
  corrLevels_.push_back("Uncorrected");
  corrLevels_.push_back("L2Relative" );
  corrLevels_.push_back("L3Absolute" );

  //open file
  TFile* file = new TFile("analyzeJetEnergyScale.root");

  // define jet energy scale histograms
  std::vector<TH1F*> jes;
  for(unsigned int idx=0; idx<corrLevels_.size(); ++idx){
    jes.push_back(new TH1F(std::string("jes").append(corrLevels_[idx]).c_str(), "Jet Response", MAXBIN, BINS));
  }

  // load base histograms
  std::vector<std::vector<TH1F*> > hists;
  for(unsigned int idx=0; idx<corrLevels_.size(); ++idx){
    std::vector<TH1F*> buffer;
    for(unsigned int jdx=0; jdx<MAXBIN; ++jdx){
      char path[50]; sprintf (path, "%s/jes_%i", corrLevels_[idx].c_str(), jdx);
      buffer.push_back((TH1F*)file->Get(path));
    }
    hists.push_back(buffer);
  }

  // fit gaussians to base histograms
  for(unsigned int idx=0; idx<corrLevels_.size(); ++idx){
    for(unsigned int jdx=0; jdx<MAXBIN; ++jdx){
      hists[idx][jdx]->Fit("gaus");
      jes[idx]->SetBinContent(jdx+1, hists[idx][jdx]->GetFunction("gaus")->GetParameter(1));
      jes[idx]->SetBinError  (jdx+1, hists[idx][jdx]->GetFunction("gaus")->GetParError (1));
    }
  }

  // setup the canvas and draw the histograms
  TCanvas* canv0 = new TCanvas("canv0", "canv0", 600, 600);
  canv0->cd(0);
  canv0->SetGridx(1);
  canv0->SetGridy(1);
  jes[2]->SetMinimum(0.);
  jes[2]->SetMaximum(2.);
  jes[2]->SetLineColor(kRed);
  jes[2]->SetLineWidth(3.);
  jes[2]->SetMarkerStyle(20.);
  jes[2]->SetMarkerColor(kRed);
  jes[2]->GetXaxis()->SetTitle("p_{T}^{gen} [GeV]");
  jes[2]->Draw();
  jes[1]->SetLineColor(kBlue);
  jes[1]->SetLineWidth(3.);
  jes[1]->SetMarkerStyle(21.);
  jes[1]->SetMarkerColor(kBlue);
  jes[1]->Draw("same");
  jes[0]->SetLineColor(kBlack);
  jes[0]->SetLineWidth(3.);
  jes[0]->SetMarkerStyle(22.);
  jes[0]->SetMarkerColor(kBlack);
  jes[0]->Draw("same");
  
  TLegend* leg = new TLegend(0.4,0.6,0.90,0.90);
  leg->SetFillStyle (0);
  leg->SetFillColor (0);
  leg->SetBorderSize(0);
  leg->AddEntry( jes[2], "L3Absolute" , "LP");
  leg->AddEntry( jes[1], "L2Relative" , "LP");
  leg->AddEntry( jes[0], "Uncorrected", "LP");
  leg->Draw("same");
}


