#ifndef PhysicsTools_Utilities_interface_LumiReWeighting_cc
#define PhysicsTools_Utilities_interface_LumiReWeighting_cc

/**
  \class    LumiReWeighting LumiReWeighting.h "PhysicsTools/Utilities/interface/LumiReWeighting.h"
  \brief    Class to provide lumi weighting for analyzers to weight "flat-to-N" MC samples to data

  This class will trivially take two histograms:
  1. The generated "flat-to-N" distributions from a given processing (or any other generated input)
  2. A histogram generated from the "estimatePileup" macro here:

  https://twiki.cern.ch/twiki/bin/view/CMS/LumiCalc#How_to_use_script_estimatePileup

  and produce weights to convert the input distribution (1) to the latter (2).

  \author Salvatore Rappoccio, modified by Mike Hildreth
  
*/
#include "TFile.h"
#include "TH1.h"
#include "TRandom1.h"
#include "TRandom2.h"
#include "TRandom3.h"
#include "TStopwatch.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>

#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "PhysicsTools/Utilities/interface/LumiReWeighting.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Common/interface/EventBase.h"

using namespace edm;

LumiReWeighting::LumiReWeighting(std::string generatedFile,
                                 std::string dataFile,
                                 std::string GenHistName,
                                 std::string DataHistName,
                                 const edm::InputTag& PileupSumInfoInputTag)
    : generatedFileName_(generatedFile),
      dataFileName_(dataFile),
      GenHistName_(GenHistName),
      DataHistName_(DataHistName),
      pileupSumInfoTag_(PileupSumInfoInputTag) {
  generatedFile_ = std::make_shared<TFile>(generatedFileName_.c_str());  //MC distribution
  dataFile_ = std::make_shared<TFile>(dataFileName_.c_str());            //Data distribution

  Data_distr_ = std::shared_ptr<TH1>((static_cast<TH1*>(dataFile_->Get(DataHistName_.c_str())->Clone())));
  MC_distr_ = std::shared_ptr<TH1>((static_cast<TH1*>(generatedFile_->Get(GenHistName_.c_str())->Clone())));

  // MC * data/MC = data, so the weights are data/MC:

  // normalize both histograms first

  Data_distr_->Scale(1.0 / Data_distr_->Integral());
  MC_distr_->Scale(1.0 / MC_distr_->Integral());

  weights_ = std::shared_ptr<TH1>(static_cast<TH1*>(Data_distr_->Clone()));

  weights_->SetName("lumiWeights");

  TH1* den = dynamic_cast<TH1*>(MC_distr_->Clone());

  //den->Scale(1.0/ den->Integral());

  weights_->Divide(den);  // so now the average weight should be 1.0

  std::cout << " Lumi/Pileup Reweighting: Computed Weights per In-Time Nint " << std::endl;

  int NBins = weights_->GetNbinsX();

  for (int ibin = 1; ibin < NBins + 1; ++ibin) {
    std::cout << "   " << ibin - 1 << " " << weights_->GetBinContent(ibin) << std::endl;
  }

  FirstWarning_ = true;
  OldLumiSection_ = -1;
}

LumiReWeighting::LumiReWeighting(const std::vector<float>& MC_distr,
                                 const std::vector<float>& Lumi_distr,
                                 const edm::InputTag& PileupSumInfoInputTag)
    : pileupSumInfoTag_(PileupSumInfoInputTag) {
  // no histograms for input: use vectors

  // now, make histograms out of them:

  // first, check they are the same size...

  if (MC_distr.size() != Lumi_distr.size()) {
    std::cerr << "ERROR: LumiReWeighting: input vectors have different sizes. Quitting... \n";
    return;
  }

  Int_t NBins = MC_distr.size();

  MC_distr_ = std::shared_ptr<TH1>(new TH1F("MC_distr", "MC dist", NBins, -0.5, float(NBins) - 0.5));
  Data_distr_ = std::shared_ptr<TH1>(new TH1F("Data_distr", "Data dist", NBins, -0.5, float(NBins) - 0.5));

  weights_ = std::shared_ptr<TH1>(new TH1F("luminumer", "luminumer", NBins, -0.5, float(NBins) - 0.5));
  TH1* den = new TH1F("lumidenom", "lumidenom", NBins, -0.5, float(NBins) - 0.5);

  for (int ibin = 1; ibin < NBins + 1; ++ibin) {
    weights_->SetBinContent(ibin, Lumi_distr[ibin - 1]);
    Data_distr_->SetBinContent(ibin, Lumi_distr[ibin - 1]);
    den->SetBinContent(ibin, MC_distr[ibin - 1]);
    MC_distr_->SetBinContent(ibin, MC_distr[ibin - 1]);
  }

  // check integrals, make sure things are normalized

  float deltaH = weights_->Integral();
  if (fabs(1.0 - deltaH) > 0.02) {  //*OOPS*...
    weights_->Scale(1.0 / weights_->Integral());
    Data_distr_->Scale(1.0 / Data_distr_->Integral());
  }
  float deltaMC = den->Integral();
  if (fabs(1.0 - deltaMC) > 0.02) {
    den->Scale(1.0 / den->Integral());
    MC_distr_->Scale(1.0 / MC_distr_->Integral());
  }

  weights_->Divide(den);  // so now the average weight should be 1.0

  std::cout << " Lumi/Pileup Reweighting: Computed Weights per In-Time Nint " << std::endl;

  for (int ibin = 1; ibin < NBins + 1; ++ibin) {
    std::cout << "   " << ibin - 1 << " " << weights_->GetBinContent(ibin) << std::endl;
  }

  FirstWarning_ = true;
  OldLumiSection_ = -1;
}

double LumiReWeighting::weight(int npv) {
  int bin = weights_->GetXaxis()->FindBin(npv);
  return weights_->GetBinContent(bin);
}

double LumiReWeighting::weight(float npv) {
  int bin = weights_->GetXaxis()->FindBin(npv);
  return weights_->GetBinContent(bin);
}

// This version of weight does all of the work for you, assuming you want to re-weight
// using the true number of interactions in the in-time beam crossing.

double LumiReWeighting::weight(const edm::EventBase& e) {
  if (FirstWarning_) {
    e.processHistory();
    //    SetFirstFalse();
    FirstWarning_ = false;
  }
  Handle<std::vector<PileupSummaryInfo> > PupInfo;
  e.getByLabel(pileupSumInfoTag_, PupInfo);

  std::vector<PileupSummaryInfo>::const_iterator PVI;

  int npv = -1;
  for (PVI = PupInfo->begin(); PVI != PupInfo->end(); ++PVI) {
    int BX = PVI->getBunchCrossing();

    if (BX == 0) {
      npv = PVI->getPU_NumInteractions();
      continue;
    }
  }

  if (npv < 0)
    std::cerr << " no in-time beam crossing found\n! ";

  return weight(npv);
}

// Use this routine to re-weight out-of-time pileup to match the in-time distribution
// As of May 2011, CMS is only sensitive to a bunch that is 50ns "late", which corresponds to
// BunchCrossing +1.  So, we use that here for re-weighting.

double LumiReWeighting::weightOOT(const edm::EventBase& e) {
  //int Run = e.run();
  int LumiSection = e.luminosityBlock();
  // do some caching here, attempt to catch file boundaries

  if (LumiSection != OldLumiSection_) {
    e.processHistory();  // keep the function call
    OldLumiSection_ = LumiSection;
  }
  // find the pileup summary information

  Handle<std::vector<PileupSummaryInfo> > PupInfo;
  e.getByLabel(pileupSumInfoTag_, PupInfo);

  std::vector<PileupSummaryInfo>::const_iterator PVI;

  int npv = -1;
  int npv50ns = -1;

  for (PVI = PupInfo->begin(); PVI != PupInfo->end(); ++PVI) {
    int BX = PVI->getBunchCrossing();

    if (BX == 0) {
      npv = PVI->getPU_NumInteractions();
    }

    if (BX == 1) {
      npv50ns = PVI->getPU_NumInteractions();
    }
  }

  // Note: for the "uncorrelated" out-of-time pileup, reweighting is only done on the 50ns
  // "late" bunch (BX=+1), since that is basically the only one that matters in terms of
  // energy deposition.

  if (npv < 0) {
    std::cerr << " no in-time beam crossing found\n! ";
    std::cerr << " Returning event weight=0\n! ";
    return 0.;
  }
  if (npv50ns < 0) {
    std::cerr << " no out-of-time beam crossing found\n! ";
    std::cerr << " Returning event weight=0\n! ";
    return 0.;
  }

  int bin = weights_->GetXaxis()->FindBin(npv);

  double inTimeWeight = weights_->GetBinContent(bin);

  double TotalWeight = 1.0;

  TotalWeight = inTimeWeight;

  return TotalWeight;
}

#endif
