#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "TH1F.h"
#include "TFile.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TLegend.h"

/// Fill the cutflow plot from 'yield' histograms
TH1F* cutflow(TFile* sample, const std::vector<std::string>& validSteps);
/// Check whether the given parameter is valid or not
bool isValid(const std::vector<std::string>& validObjects, const std::string& value);

/// -------------------------------------------------------------------------------
///
/// Display the selection monitoring plots. The parameters are the histogram name 
/// withing the root file and the selection step that make up the corresponding 
/// directory within the root file. All valid parameter are given in the vectors 
/// validHists_ (for the hostogram names) and validSteps_ (for the selection steps).
/// The use case is:
/// .x PhysicsTools/PatExamples/bin/monitorTopSelection.C+("muonPt", "Step1")
///
/// -------------------------------------------------------------------------------
void monitorTopSelection(const std::string& histName="yield", std::string selectionStep="Step1")
{
  gStyle->SetOptStat(0);

  // list of valid histogram names
  std::vector<std::string> validHists_;
  validHists_.push_back("yield"   );
  validHists_.push_back("elecMult");
  validHists_.push_back("elecIso" );
  validHists_.push_back("elecPt"  );
  validHists_.push_back("muonMult");
  validHists_.push_back("muonIso" );
  validHists_.push_back("muonPt"  );
  validHists_.push_back("jetMult" );
  validHists_.push_back("jet1Pt"  );
  validHists_.push_back("jet2Pt"  );
  validHists_.push_back("jet3Pt"  );
  validHists_.push_back("jet4Pt"  );
  validHists_.push_back("met"     );
  
  // list of valid selection steps
  std::vector<std::string> validSteps_;
  validSteps_.push_back("Step1"   );
  validSteps_.push_back("Step2"   );
  validSteps_.push_back("Step3a"  );
  validSteps_.push_back("Step4"   );
  validSteps_.push_back("Step5"   );
  validSteps_.push_back("Step6a"  );
  validSteps_.push_back("Step6b"  );
  validSteps_.push_back("Step6c"  );
  validSteps_.push_back("Step7"   );
  
  // check validity of input
  if(!isValid(validHists_, histName) || !isValid(validSteps_, selectionStep)){ 
    return;
  }
  
  // list of input samples (ATTENTION obey order)
  std::vector<string> samples_;
  samples_.push_back("analyzePatTopSelection_ttbar.root");
  samples_.push_back("analyzePatTopSelection_wjets.root");
  samples_.push_back("analyzePatTopSelection_zjets.root");
  samples_.push_back("analyzePatTopSelection_qcd.root"  );
  samples_.push_back("analyzePatTopSelection.root");

  //open files
  std::vector<TFile*> files;
  for(unsigned int idx=0; idx<samples_.size(); ++idx){
    files.push_back(new TFile(samples_[idx].c_str()));
  }
  // load histograms
  std::vector<TH1F*> hists;
  if(histName=="yield"){
    for(unsigned int idx=0; idx<files.size(); ++idx){
      hists.push_back(cutflow(files[idx], validSteps_));
    }
  } 
  else{
    std::string histPath = selectionStep.append(std::string("/").append(histName));
    for(unsigned int idx=0; idx<files.size(); ++idx){
      hists.push_back((TH1F*)files[idx]->Get((std::string("mon").append(histPath)).c_str()));
    }
  }
  float lumi = 2.;
  // scale and stack histograms for simulated events
  // scales for 1pb-1: ttbar,  wjets,  zjets,  qcd 
  float scales[] =    {0.165,  0.312,  0.280, 0.287};
  for(unsigned int idx=0; idx<samples_.size()-1; ++idx){
    hists[idx]->Scale(lumi*scales[idx]);
  }
  for(unsigned int idx=1; idx<samples_.size()-1; ++idx){
    hists[idx]->Add(hists[idx-1]);
  }
  // setup the canvas and draw the histograms
  TCanvas* canv0 = new TCanvas("canv0", "canv0", 600, 600);
  canv0->cd(0);
  canv0->SetLogy(1);
  if(histName=="yield"){
    hists[3]->SetTitle("Selection Steps");
  }
  hists[3]->SetMinimum(1.);
  hists[3]->SetMaximum(20.*hists[4]->GetMaximum());
  hists[3]->SetFillColor(kYellow);
  hists[3]->Draw();
  hists[2]->SetFillColor(kAzure-2);
  hists[2]->Draw("same");
  hists[1]->SetFillColor(kGreen-3);
  hists[1]->Draw("same");
  hists[0]->SetFillColor(kRed+1);
  hists[0]->Draw("same");
  // plot data points
  hists[4]->SetLineWidth(3.);
  hists[4]->SetLineColor(kBlack);
  hists[4]->SetMarkerColor(kBlack);
  hists[4]->SetMarkerStyle(20.);
  hists[4]->Draw("esame");
  canv0->RedrawAxis();
  
  TLegend* leg = new TLegend(0.35,0.6,0.85,0.90);
  leg->SetFillStyle ( 0);
  leg->SetFillColor ( 0);
  leg->SetBorderSize( 0);
  leg->AddEntry( hists[4], "CMS Data 2010 (2 pb^{-1})"     , "PL");
  leg->AddEntry( hists[3], "QCD"                           , "F");
  leg->AddEntry( hists[2], "Z/#gamma#rightarrowl^{+}l^{-}" , "F");
  leg->AddEntry( hists[1], "W#rightarrowl#nu"              , "F");
  leg->AddEntry( hists[0], "t#bar{t} (incl)"               , "F");
  leg->Draw("same");
}

TH1F* cutflow(TFile* sample, const std::vector<std::string>& validSteps)
{
  // book histogram
  TH1F* hist = new TH1F(sample->GetName(), sample->GetName(), validSteps.size(), 0., validSteps.size());
  // set labels
  for(unsigned int idx=0; idx<validSteps.size(); ++idx){
    hist->GetXaxis()->SetBinLabel( idx+1 , validSteps[idx].c_str());
  }
  hist->LabelsOption("h", "X"); //"h", "v", "u", "d"
  // fill histogram
  for(unsigned int idx=0; idx<validSteps.size(); ++idx){
    TH1F* buffer = (TH1F*)sample->Get((std::string("mon").append(validSteps[idx]).append("/yield")).c_str());
    hist->SetBinContent(idx+1, buffer->GetBinContent(1));
  }
  return hist;
}

bool isValid(const std::vector<std::string>& validObjects, const std::string& value)
{
  // check whether value is in the list of valid hostogram names
  if(std::find(validObjects.begin(), validObjects.end(), value)==validObjects.end()){
    std::cout << " ERROR : " << value << " is not a valid value" << std::endl;
    std::cout << " List of valid values:" << std::endl;
    for(std::vector<std::string>::const_iterator obj=validObjects.begin(); obj!=validObjects.end(); ++obj){
      std::cout << " " << (*obj) << std::endl;
    }
    std::cout << std::endl;
    return false;
  }
  return true;
}
