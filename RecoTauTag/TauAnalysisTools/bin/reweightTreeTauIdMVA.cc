
/** \executable reweightTreeTauIdMVA
 *
 * Compute reweighting factors to make distribution of signal and background events 
 * used to train MVA for identifying hadronic tau decays flat in Pt and/or eta.
 *
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: trainTauIdMVA.cc,v 1.1 2012/03/06 17:34:42 veelken Exp $
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "DataFormats/FWLite/interface/InputSource.h"
#include "DataFormats/FWLite/interface/OutputFiles.h"

#include "RecoTauTag/TauAnalysisTools/bin/tauIdMVATrainingAuxFunctions.h"

#include <TFile.h>
#include <TChain.h>
#include <TTree.h>
#include <TBenchmark.h>
#include <TH1.h>
#include <TH2.h>
#include <TString.h>
#include <TObjArray.h>
#include <TObjString.h>

#include <iostream>
#include <string>
#include <vector>
#include <assert.h>

typedef std::vector<std::string> vstring;

enum { kUndefined, kSaveSignal, kSaveBackground };

namespace
{
  void normalizeHistogram(TH1* histogram)
  {
    if ( histogram->Integral() > 0. ) {
      if ( !histogram->GetSumw2N() ) histogram->Sumw2();
      histogram->Scale(1./histogram->Integral());
    } 
  }
  
  void fillHistogramsForPtVsEtaReweighting(TTree* inputTree, 
					   const std::string& branchNamePt, const std::string& branchNameEta, const std::string& branchNameEvtWeight, 
					   TH1* histogramLogPt, TH1* histogramAbsEta, TH2* histogramLogPtVsAbsEta,
					   unsigned reportEvery)
  {
    Float_t pt;
    inputTree->SetBranchAddress(branchNamePt.data(), &pt);
    Float_t eta;
    inputTree->SetBranchAddress(branchNameEta.data(), &eta);
    
    Float_t evtWeight = 1.0;
    if ( branchNameEvtWeight != "" ) {
      inputTree->SetBranchAddress(branchNameEvtWeight.data(), &evtWeight);
    }
    
    int numEntries = inputTree->GetEntries();
    for ( int iEntry = 0; iEntry < numEntries; ++iEntry ) {
      if ( iEntry > 0 && (iEntry % reportEvery) == 0 ) {
	std::cout << "processing Entry " << iEntry << std::endl;
      }
      
      inputTree->GetEntry(iEntry);
      
      Float_t absEta = TMath::Abs(eta);
      Float_t logPt = TMath::Log(TMath::Max((Float_t)1., pt));
      
      histogramLogPt->Fill(logPt, evtWeight);
      histogramAbsEta->Fill(absEta, evtWeight);
      histogramLogPtVsAbsEta->Fill(absEta, logPt, evtWeight);
    }

    normalizeHistogram(histogramLogPt);
    normalizeHistogram(histogramAbsEta);
    normalizeHistogram(histogramLogPtVsAbsEta);
  }

  void makeFlatHistogram(TH1* histogram)
  {
    int numBinsX = histogram->GetNbinsX();
    int numBinsY = histogram->GetNbinsY();
    double binContent = 1./(numBinsX*numBinsY);
    for ( int iBinX = 1; iBinX <= numBinsX; ++iBinX ) {
      for ( int iBinY = 1; iBinY <= numBinsY; ++iBinY ) {
	histogram->SetBinContent(iBinX, iBinY, binContent);
      }
    }
  }

  void makeMinHistogram(TH1* histogram_min, const TH1* histogram_signal, const TH1* histogram_background)
  {
    assert(histogram_min->GetNbinsX() == histogram_signal->GetNbinsX());
    assert(histogram_min->GetNbinsX() == histogram_background->GetNbinsX());
    int numBinsX = histogram_min->GetNbinsX();

    assert(histogram_min->GetNbinsY() == histogram_signal->GetNbinsY());
    assert(histogram_min->GetNbinsY() == histogram_background->GetNbinsY());
    int numBinsY = histogram_min->GetNbinsY();

    for ( int iBinX = 1; iBinX <= numBinsX; ++iBinX ) {
      for ( int iBinY = 1; iBinY <= numBinsY; ++iBinY ) {
	double binContent_signal = histogram_signal->GetBinContent(iBinX, iBinY);
	double binError_signal = histogram_signal->GetBinError(iBinX, iBinY);
	double binContent_background = histogram_background->GetBinContent(iBinX, iBinY);
	double binError_background = histogram_background->GetBinError(iBinX, iBinY);
	if ( binContent_signal < binContent_background ) {
	  histogram_min->SetBinContent(iBinX, iBinY, binContent_signal);
	  histogram_min->SetBinError(iBinX, iBinY, binError_signal);
	} else {
	  histogram_min->SetBinContent(iBinX, iBinY, binContent_background);
	  histogram_min->SetBinError(iBinX, iBinY, binError_background);
	}
      }
    }
  }

  TH1* divideHistograms(const TH1* numerator, const TH1* denominator)
  {
    std::string histogramName_ratio = Form("%s_div_%s", numerator->GetName(), denominator->GetName());
    TH1* histogram_ratio = (TH1*)numerator->Clone(histogramName_ratio.data());
    histogram_ratio->Divide(denominator);
    return histogram_ratio;
  }
}

int main(int argc, char* argv[]) 
{
//--- parse command-line arguments
  if ( argc < 2 ) {
    std::cout << "Usage: " << argv[0] << " [parameters.py]" << std::endl;
    return 0;
  }

  std::cout << "<reweightTreeTauIdMVA>:" << std::endl;

//--- keep track of time it takes the macro to execute
  TBenchmark clock;
  clock.Start("reweightTreeTauIdMVA");

//--- CV: disable automatic association of histograms to files
//       (default behaviour ROOT object ownership, 
//        cf. http://root.cern.ch/download/doc/Users_Guide_5_26.pdf section 8)
  TH1::AddDirectory(false);

//--- read python configuration parameters
  if ( !edm::readPSetsFrom(argv[1])->existsAs<edm::ParameterSet>("process") ) 
    throw cms::Exception("reweightTreeTauIdMVA") 
      << "No ParameterSet 'process' found in configuration file = " << argv[1] << " !!\n";

  edm::ParameterSet cfg = edm::readPSetsFrom(argv[1])->getParameter<edm::ParameterSet>("process");

  edm::ParameterSet cfgReweightTreeTauIdMVA = cfg.getParameter<edm::ParameterSet>("reweightTreeTauIdMVA");
  
  std::string inputTreeName = cfgReweightTreeTauIdMVA.getParameter<std::string>("inputTreeName");
  std::string outputTreeName = cfgReweightTreeTauIdMVA.getParameter<std::string>("outputTreeName");
  
  vstring signalSamples = cfgReweightTreeTauIdMVA.getParameter<vstring>("signalSamples");
  vstring backgroundSamples = cfgReweightTreeTauIdMVA.getParameter<vstring>("backgroundSamples");
  
  bool applyPtReweighting = cfgReweightTreeTauIdMVA.getParameter<bool>("applyPtReweighting");
  std::string branchNamePt = cfgReweightTreeTauIdMVA.getParameter<std::string>("branchNamePt");
  bool applyEtaReweighting = cfgReweightTreeTauIdMVA.getParameter<bool>("applyEtaReweighting");
  std::string branchNameEta = cfgReweightTreeTauIdMVA.getParameter<std::string>("branchNameEta");
  TString reweightOption_tstring = cfgReweightTreeTauIdMVA.getParameter<std::string>("reweight").data();  
  int reweight_or_KILL = kReweight;
  int reweightOption = -1;
  TObjArray* reweightOption_items = reweightOption_tstring.Tokenize(":");
  int numItems = reweightOption_items->GetEntries();
  for ( int iItem = 0; iItem < numItems; ++iItem ) {
    TObjString* item = dynamic_cast<TObjString*>(reweightOption_items->At(iItem));
    assert(item);
    std::string item_string = item->GetString().Data();
    if      ( item_string == "none"       ) continue;
    else if ( item_string == "signal"     ) reweightOption   = kReweight_or_KILLsignal;
    else if ( item_string == "background" ) reweightOption   = kReweight_or_KILLbackground;
    else if ( item_string == "flat"       ) reweightOption   = kReweight_or_KILLflat;
    else if ( item_string == "min"        ) reweightOption   = kReweight_or_KILLmin;
    else if ( item_string == "KILL"       ) reweight_or_KILL = kKILL;    
    else throw cms::Exception("reweightTreeTauIdMVA") 
      << "Invalid Configuration parameter 'reweight' = " << reweightOption_tstring.Data() << " !!\n";
  }

  vstring inputVariables = cfgReweightTreeTauIdMVA.getParameter<vstring>("inputVariables");

  vstring spectatorVariables = cfgReweightTreeTauIdMVA.getParameter<vstring>("spectatorVariables");

  std::string branchNameEvtWeight = cfgReweightTreeTauIdMVA.getParameter<std::string>("branchNameEvtWeight");

  bool keepAllBranches = cfgReweightTreeTauIdMVA.getParameter<bool>("keepAllBranches");
  bool checkBranchesForNaNs = cfgReweightTreeTauIdMVA.getParameter<bool>("checkBranchesForNaNs");

  fwlite::InputSource inputFiles(cfg); 
  int maxEvents = inputFiles.maxEvents();
  std::cout << " maxEvents = " << maxEvents << std::endl;
  unsigned reportEvery = inputFiles.reportAfter();

  std::string outputFileName = cfgReweightTreeTauIdMVA.getParameter<std::string>("outputFileName");
  std::cout << " outputFileName = " << outputFileName << std::endl;

  std::string saveOption_string = cfgReweightTreeTauIdMVA.getParameter<std::string>("save");
  int saveOption = -1;
  if      ( saveOption_string == "signal"     ) saveOption = kSaveSignal;
  else if ( saveOption_string == "background" ) saveOption = kSaveBackground;
  else throw cms::Exception("reweightTreeTauIdMVA") 
    << "Invalid Configuration parameter 'save' = " << saveOption_string << " !!\n";

  TChain* inputTree_signal = new TChain(inputTreeName.data());
  TChain* inputTree_background = new TChain(inputTreeName.data());
  for ( vstring::const_iterator inputFileName = inputFiles.files().begin();
	inputFileName != inputFiles.files().end(); ++inputFileName ) {
    bool matchesSample_signal = false;
    for ( vstring::const_iterator signal = signalSamples.begin();
	  signal != signalSamples.end(); ++signal ) {
      if ( inputFileName->find(*signal) != std::string::npos ) matchesSample_signal = true;
    }
    bool matchesSample_background = false;
    for ( vstring::const_iterator background = backgroundSamples.begin();
	  background != backgroundSamples.end(); ++background ) {
      if ( inputFileName->find(*background) != std::string::npos ) matchesSample_background = true;
    }
    if ( (matchesSample_signal && matchesSample_background) || !(matchesSample_signal || matchesSample_background) ) {
      throw cms::Exception("reweightTreeTauIdMVA") 
	<< "Failed to identify if inputFile = " << (*inputFileName) << " is signal or background !!\n";
    }
    if ( matchesSample_signal ) {
      std::cout << "input Tree for signal: adding file = " << (*inputFileName) << std::endl;
      inputTree_signal->AddFile(inputFileName->data());
    } 
    if ( matchesSample_background ) {
      std::cout << "input Tree for background: adding file = " << (*inputFileName) << std::endl;
      inputTree_background->AddFile(inputFileName->data());
    } 
  }
  
  if ( !(inputTree_signal->GetListOfFiles()->GetEntries() >= 1) ) {
    throw cms::Exception("reweightTreeTauIdMVA") 
      << "Failed to identify input Tree for signal !!\n";
  }
  if ( !(inputTree_background->GetListOfFiles()->GetEntries() >= 1) ) {
    throw cms::Exception("reweightTreeTauIdMVA") 
      << "Failed to identify input Tree for background !!\n";
  }

  vstring branchesToKeep_expressions = inputVariables;
  branchesToKeep_expressions.push_back(branchNameEvtWeight);
  branchesToKeep_expressions.push_back(branchNamePt);
  branchesToKeep_expressions.push_back(branchNameEta);
  branchesToKeep_expressions.insert(branchesToKeep_expressions.end(), spectatorVariables.begin(), spectatorVariables.end());

  if ( keepAllBranches ) {
    TTree* inputTree = 0;
    if      ( saveOption == kSaveSignal     ) inputTree = inputTree_signal;
    else if ( saveOption == kSaveBackground ) inputTree = inputTree_background;
    if ( inputTree ) {
      TObjArray* branches = inputTree->GetListOfBranches();
      int numBranches = branches->GetEntries();
      for ( int iBranch = 0; iBranch < numBranches; ++iBranch ) {
	TBranch* branch = dynamic_cast<TBranch*>(branches->At(iBranch));
	assert(branch);
	std::string branchName = branch->GetName();
	branchesToKeep_expressions.push_back(branchName);
      }
    }
  }

//--- check if signal or background entries are to be reweighted such that
//    Pt and eta distributions of tau candidates match
  if ( applyPtReweighting || applyEtaReweighting ) {
    TH1* histogramLogPt_signal             = new TH1D("histogramLogPt_signal",             "log(tauPt) signal",                     400, 0., 10.);
    TH1* histogramLogPt_background         = new TH1D("histogramLogPt_background",         "log(tauPt) background",                 400, 0., 10.);
    TH1* histogramAbsEta_signal            = new TH1D("histogramAbsEta_signal",            "abs(tauEta) signal",                    100, 0.,  5.);
    TH1* histogramAbsEta_background        = new TH1D("histogramAbsEta_background",        "abs(tauEta) background",                100, 0.,  5.);
    TH2* histogramLogPtVsAbsEta_signal     = new TH2D("histogramLogPtVsAbsEta_signal",     "log(tauPt) vs. abs(tauEta) signal",     100, 0.,  5., 400, 0., 10.);
    TH2* histogramLogPtVsAbsEta_background = new TH2D("histogramLogPtVsAbsEta_background", "log(tauPt) vs. abs(tauEta) background", 100, 0.,  5., 400, 0., 10.);
    if      ( applyPtReweighting && applyEtaReweighting ) std::cout << "Info: filling histogram for pT vs. eta-reweighting" << std::endl;
    else if ( applyPtReweighting                        ) std::cout << "Info: filling histogram for pT-reweighting" << std::endl;
    else if (                       applyEtaReweighting ) std::cout << "Info: filling histogram for eta-reweighting" << std::endl;
    fillHistogramsForPtVsEtaReweighting(
      inputTree_signal, 
      branchNamePt, branchNameEta, branchNameEvtWeight, 
      histogramLogPt_signal, histogramAbsEta_signal, histogramLogPtVsAbsEta_signal,
      reportEvery);
    fillHistogramsForPtVsEtaReweighting(
      inputTree_background, 
      branchNamePt, branchNameEta, branchNameEvtWeight, 
      histogramLogPt_background, histogramAbsEta_background, histogramLogPtVsAbsEta_background,
      reportEvery);

    TH1* histogramLogPt_reweight_signal             = 0;
    TH1* histogramAbsEta_reweight_signal            = 0;
    TH2* histogramLogPtVsAbsEta_reweight_signal     = 0;
    TH1* histogramLogPt_reweight_background         = 0;
    TH1* histogramAbsEta_reweight_background        = 0;
    TH2* histogramLogPtVsAbsEta_reweight_background = 0;
    if ( reweightOption == kReweight_or_KILLsignal ) {
      histogramLogPt_reweight_signal             = divideHistograms(histogramLogPt_background, histogramLogPt_signal);
      histogramAbsEta_reweight_signal            = divideHistograms(histogramAbsEta_background, histogramAbsEta_signal);
      histogramLogPtVsAbsEta_reweight_signal     = dynamic_cast<TH2*>(divideHistograms(histogramLogPtVsAbsEta_background, histogramLogPtVsAbsEta_signal));
    } else if ( reweightOption == kReweight_or_KILLbackground ) {
      histogramLogPt_reweight_background         = divideHistograms(histogramLogPt_signal, histogramLogPt_background);
      histogramAbsEta_reweight_background        = divideHistograms(histogramAbsEta_signal, histogramAbsEta_background);
      histogramLogPtVsAbsEta_reweight_background = dynamic_cast<TH2*>(divideHistograms(histogramLogPtVsAbsEta_signal, histogramLogPtVsAbsEta_background));
    } else if ( reweightOption == kReweight_or_KILLflat ) {
      TH1* histogramLogPt_flat         = new TH1D("histogramLogPt_flat",         "log(tauPt) flat",                 400, 0., 10.);
      TH1* histogramAbsEta_flat        = new TH1D("histogramAbsEta_flat",        "abs(tauEta) flat",                100, 0.,  5.);
      TH2* histogramLogPtVsAbsEta_flat = new TH2D("histogramLogPtVsAbsEta_flat", "log(tauPt) vs. abs(tauEta) flat", 100, 0.,  5., 400, 0., 10.);
      makeFlatHistogram(histogramLogPt_flat);
      makeFlatHistogram(histogramAbsEta_flat);
      makeFlatHistogram(histogramLogPtVsAbsEta_flat);
      histogramLogPt_reweight_signal             = divideHistograms(histogramLogPt_flat, histogramLogPt_signal);
      histogramAbsEta_reweight_signal            = divideHistograms(histogramAbsEta_flat, histogramAbsEta_signal);
      histogramLogPtVsAbsEta_reweight_signal     = dynamic_cast<TH2*>(divideHistograms(histogramLogPtVsAbsEta_flat, histogramLogPtVsAbsEta_signal));
      histogramLogPt_reweight_background         = divideHistograms(histogramLogPt_flat, histogramLogPt_background);
      histogramAbsEta_reweight_background        = divideHistograms(histogramAbsEta_flat, histogramAbsEta_background);
      histogramLogPtVsAbsEta_reweight_background = dynamic_cast<TH2*>(divideHistograms(histogramLogPtVsAbsEta_flat, histogramLogPtVsAbsEta_background));
      delete histogramLogPt_flat;
      delete histogramAbsEta_flat;
      delete histogramLogPtVsAbsEta_flat;
    } else if ( reweightOption == kReweight_or_KILLmin ) {
      TH1* histogramLogPt_min         = new TH1D("histogramLogPt_min",         "log(tauPt) min(signal, background)",                 400, 0., 10.);
      TH1* histogramAbsEta_min        = new TH1D("histogramAbsEta_min",        "abs(tauEta) min(signal, background)",                100, 0.,  5.);
      TH2* histogramLogPtVsAbsEta_min = new TH2D("histogramLogPtVsAbsEta_min", "log(tauPt) vs. abs(tauEta) min(signal, background)", 100, 0.,  5., 400, 0., 10.);
      makeMinHistogram(histogramLogPt_min, histogramLogPt_signal, histogramLogPt_background);
      makeMinHistogram(histogramAbsEta_min, histogramAbsEta_signal, histogramAbsEta_background);
      makeMinHistogram(histogramLogPtVsAbsEta_min, histogramLogPtVsAbsEta_signal, histogramLogPtVsAbsEta_background);
      histogramLogPt_reweight_signal             = divideHistograms(histogramLogPt_min, histogramLogPt_signal);
      histogramAbsEta_reweight_signal            = divideHistograms(histogramAbsEta_min, histogramAbsEta_signal);
      histogramLogPtVsAbsEta_reweight_signal     = dynamic_cast<TH2*>(divideHistograms(histogramLogPtVsAbsEta_min, histogramLogPtVsAbsEta_signal));
      histogramLogPt_reweight_background         = divideHistograms(histogramLogPt_min, histogramLogPt_background);
      histogramAbsEta_reweight_background        = divideHistograms(histogramAbsEta_min, histogramAbsEta_background);
      histogramLogPtVsAbsEta_reweight_background = dynamic_cast<TH2*>(divideHistograms(histogramLogPtVsAbsEta_min, histogramLogPtVsAbsEta_background));
      delete histogramLogPt_min;
      delete histogramAbsEta_min;
      delete histogramLogPtVsAbsEta_min;
    } 

    TFile* outputFile = new TFile(outputFileName.data(), "RECREATE");
    if ( saveOption == kSaveSignal ) {
      bool reweightSignal = (reweightOption == kReweight_or_KILLsignal || reweightOption == kReweight_or_KILLflat || reweightOption == kReweight_or_KILLmin);
      bool applyPtReweighting_signal  = (applyPtReweighting  && reweightSignal);
      bool applyEtaReweighting_signal = (applyEtaReweighting && reweightSignal);
      TTree* outputTree_signal = preselectTree(
	inputTree_signal, outputTreeName, 
	"", branchesToKeep_expressions, 
	0, branchNamePt, branchNameEta, "",
	reweight_or_KILL, applyPtReweighting_signal, applyEtaReweighting_signal, histogramLogPt_reweight_signal, histogramAbsEta_reweight_signal, histogramLogPtVsAbsEta_reweight_signal,
	maxEvents, checkBranchesForNaNs, reportEvery);
      std::cout << "--> output Tree for signal contains " << outputTree_signal->GetEntries() << " Entries." << std::endl;

      //std::cout << "output Tree:" << std::endl;
      //outputTree_signal->Print();
      //outputTree_signal->Scan("*", "", "", 20, 0);

      std::cout << "writing output Tree to file = " << outputFileName << "." << std::endl;
      outputTree_signal->Write(); // CV: **not** done automatically by ROOT when outputFile gets closed
                                  //    (unless Tree is written to file explicitely, some entries may be missing, 
                                  //     which happened to be the most interesting high Pt events that were processed at the end of the job !!)
    }
    if ( saveOption == kSaveBackground ) {
      bool reweightBackground = (reweightOption == kReweight_or_KILLbackground || reweightOption == kReweight_or_KILLflat || reweightOption == kReweight_or_KILLmin);
      bool applyPtReweighting_background  = (applyPtReweighting  && reweightBackground);
      bool applyEtaReweighting_background = (applyEtaReweighting && reweightBackground);
      TTree* outputTree_background = preselectTree(
	inputTree_background, outputTreeName, 
	"", branchesToKeep_expressions, 
	0, branchNamePt, branchNameEta, "",
	reweight_or_KILL, applyPtReweighting_background, applyEtaReweighting_background, histogramLogPt_reweight_background, histogramAbsEta_reweight_background, histogramLogPtVsAbsEta_reweight_background,
	maxEvents, checkBranchesForNaNs, reportEvery);
      std::cout << "--> output Tree for background contains " << outputTree_background->GetEntries() << " Entries." << std::endl;
  
      //std::cout << "output Tree:" << std::endl;
      //outputTree_background->Print();
      //outputTree_background->Scan("*", "", "", 20, 0);

      std::cout << "writing output Tree to file = " << outputFileName << "." << std::endl;
      outputTree_background->Write(); // CV: **not** done automatically by ROOT when outputFile gets closed
                                      //    (unless Tree is written to file explicitely, some entries may be missing, 
                                      //     which happened to be the most interesting high Pt events that were processed at the end of the job !!)
    }
    delete outputFile;

    size_t idx = outputFileName.find_last_of('.');
    std::string outputFileName_histograms = std::string(outputFileName, 0, idx);
    outputFileName_histograms.append("_histograms");
    if ( idx != std::string::npos ) outputFileName_histograms.append(std::string(outputFileName, idx));
    std::cout << "outputFileName_histograms = " << outputFileName_histograms << std::endl;
    TFile* outputFile_histograms = new TFile(outputFileName_histograms.data(), "RECREATE");
    if ( histogramLogPt_signal                      ) histogramLogPt_signal->Write();
    if ( histogramAbsEta_signal                     ) histogramAbsEta_signal->Write();
    if ( histogramLogPtVsAbsEta_signal              ) histogramLogPtVsAbsEta_signal->Write();
    if ( histogramLogPt_reweight_signal             ) histogramLogPt_reweight_signal->Write();
    if ( histogramAbsEta_reweight_signal            ) histogramAbsEta_reweight_signal->Write();
    if ( histogramLogPtVsAbsEta_signal              ) histogramLogPtVsAbsEta_reweight_signal->Write();
    if ( histogramLogPt_background                  ) histogramLogPt_background->Write();
    if ( histogramAbsEta_background                 ) histogramAbsEta_background->Write();
    if ( histogramLogPtVsAbsEta_background          ) histogramLogPtVsAbsEta_background->Write();
    if ( histogramLogPt_reweight_background         ) histogramLogPt_reweight_background->Write();
    if ( histogramAbsEta_reweight_background        ) histogramAbsEta_reweight_background->Write();
    if ( histogramLogPtVsAbsEta_reweight_background ) histogramLogPtVsAbsEta_reweight_background->Write();
    delete outputFile_histograms;

    delete histogramLogPt_signal;
    delete histogramLogPt_background;
    delete histogramLogPt_reweight_signal;
    delete histogramLogPt_reweight_background;
    delete histogramAbsEta_signal;
    delete histogramAbsEta_background;
    delete histogramAbsEta_reweight_signal;
    delete histogramAbsEta_reweight_background;
    delete histogramLogPtVsAbsEta_signal;
    delete histogramLogPtVsAbsEta_background;
    delete histogramLogPtVsAbsEta_reweight_signal;
    delete histogramLogPtVsAbsEta_reweight_background;
  }

  delete inputTree_signal;
  delete inputTree_background;
  
  clock.Show("reweightTreeTauIdMVA");

  return 0;
}
