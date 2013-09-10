
/** \executable computeWPcutsAntiElectronDiscrMVA
 *
 * Compute 16-dimensional cuts definining Loose, Medium, Tight and VTight working-points of anti-electron MVA4 discriminator
 *
 * \author Christian Veelken, LLR;
 *         Ivo Naranjo, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: computeWPcutsAntiElectronDiscrMVA.cc,v 1.1 2012/03/06 17:34:42 veelken Exp $
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "DataFormats/FWLite/interface/InputSource.h"
#include "DataFormats/FWLite/interface/OutputFiles.h"

#include <TFile.h>
#include <TChain.h>
#include <TTree.h>
#include <TTreeFormula.h>
#include <TObjArray.h>
#include <TBranch.h>
#include <TString.h>
#include <TMath.h>
#include <TBenchmark.h>

#include <iostream>
#include <string>
#include <vector>
#include <assert.h>

void normalizeHistogram(TH1* histogram)
{
  if ( histogram && histogram->Integral() > 0. ) {
    histogram->Scale(1./histogram->Integral());
  }
}

double compIntegral(const TH1* histogram, double cut)
{
  double integral = 0.;
  if ( histogram ) {
    int numBins = histogram->GetNbinsX();
    for ( int iBin = numBins; iBin >= 1; --iBin ) {
      if ( histogram->GetBinCenter(iBin) > cut ) {
	integral += histogram->GetBinContent(iBin);
      } else {
	break;
      }
    }
  }
  return integral;
}

struct workingPointEntryType
{
  double targetSignalEfficiency_;
  double minPt_;
  double maxPt_;
  typedef std::map<Int_t, double> wpMap; // key = category
  wpMap cuts_;
  double S_;
  double B_;
  double SoverB_;
};

workingPointEntryType computeDifferentialWP(double targetSignalEfficiency, 
					    const std::vector<int>& categories, 
					    std::map<Int_t, TH1*>& histograms_mvaOutput_signal, std::map<Int_t, double>& categoryProbabilities_signal,
					    std::map<Int_t, TH1*>& histograms_mvaOutput_background, std::map<Int_t, double>& categoryProbabilities_background)
{
  std::cout << "<computeDifferentialWP>:" << std::endl;
  std::cout << " targetSignalEfficiency = " << targetSignalEfficiency << std::endl;
  
  workingPointEntryType workingPoint;
  workingPoint.targetSignalEfficiency_ = targetSignalEfficiency;
  for ( std::vector<int>::const_iterator category = categories.begin();
	category != categories.end(); ++category ) {
    workingPoint.cuts_[*category] = 1.;
  }

  double Ssum = 0.;
  double Bsum = 0.;
  std::map<Int_t, double> dSdOutput; // key = category
  std::map<Int_t, double> dBdOutput; // key = category
  const double stepSize = 1.e-3; // needs to be multiple of bin-width
  const double windowSize = 1.e-2;
  int step = 0;
  while ( Ssum <= targetSignalEfficiency ) {
    // CV: compute how much signal yield and background yield change
    //     in case cut on BDT output in each category is lowered by stepsize
    for ( std::vector<int>::const_iterator category = categories.begin();
	  category != categories.end(); ++category ) {
      dSdOutput[*category] = 
        TMath::Abs((compIntegral(histograms_mvaOutput_signal[*category], workingPoint.cuts_[*category]) 
                  - compIntegral(histograms_mvaOutput_signal[*category], workingPoint.cuts_[*category] - windowSize))/windowSize);
      dBdOutput[*category] = 
        TMath::Abs((compIntegral(histograms_mvaOutput_background[*category], workingPoint.cuts_[*category]) 
                  - compIntegral(histograms_mvaOutput_background[*category], workingPoint.cuts_[*category] - windowSize))/windowSize);
    }

    // CV: protection against case that BDT output for signal has no entries between current cutValue and (cutValue - stepsize) for any category,
    //     in that case lower cutValue by stepsize for all categories
    bool hasSignal = false;
    for ( std::vector<int>::const_iterator category = categories.begin();
	  category != categories.end(); ++category ) {
      if ( dSdOutput[*category] > 0. ) {
	hasSignal = true;
	break;
      }
    }
    if ( !hasSignal ) {
      for ( std::vector<int>::const_iterator category = categories.begin();
	    category != categories.end(); ++category ) {
	workingPoint.cuts_[*category] -= stepSize;
      }
      continue;
    }

    // CV: lower cut on BDT output in the category in which background yield 
    //     increases by the least amount per unit of increase in signal yield
    int bestCategory = -1;
    double dSoverBmax = 0.;
    for ( std::vector<int>::const_iterator category = categories.begin();
	  category != categories.end(); ++category ) {
      double dSoverB = (categoryProbabilities_signal[*category]*dSdOutput[*category])/(categoryProbabilities_background[*category]*dBdOutput[*category]);
      if ( dSoverB > dSoverBmax ) {
	dSoverBmax = dSoverB;
	bestCategory = (*category);
      }
    }
    assert(bestCategory != -1);

    // CV: update cutValue
    workingPoint.cuts_[bestCategory] -= stepSize;

    // CV: update total signal and background yields corresponding to current cutValue
    Ssum = 0.;
    Bsum = 0.;
    for ( std::vector<int>::const_iterator category = categories.begin();
	  category != categories.end(); ++category ) {
      Ssum += (categoryProbabilities_signal[*category]*compIntegral(histograms_mvaOutput_signal[*category], workingPoint.cuts_[*category]));
      Bsum += (categoryProbabilities_background[*category]*compIntegral(histograms_mvaOutput_background[*category], workingPoint.cuts_[*category]));
    }
    
    if ( (step % 1000) == 0 ){ 
      std::cout << "step #" << step << ": Ssum = " << Ssum << ", Bsum = " << Bsum << std::endl;
      for ( std::vector<int>::const_iterator category = categories.begin();
	    category != categories.end(); ++category ) {
	std::cout << " category #"<< (*category) << ": cut = " << workingPoint.cuts_[*category] << std::endl;
      }
    }
    ++step;
  }

  std::cout << "S = " << Ssum << ", B = " << Bsum << " --> S/B = "<< (Ssum/Bsum) << std::endl;
  for ( std::vector<int>::const_iterator category = categories.begin();
	category != categories.end(); ++category ) {
    std::cout << " category #"<< (*category) << ": cut = " << workingPoint.cuts_[*category] << std::endl;
  }

  workingPoint.S_ = Ssum;
  workingPoint.B_ = Bsum;
  workingPoint.SoverB_ = ( Bsum > 0. ) ? (Ssum/Bsum) : 1.e+6;

  return workingPoint;
}

void testWP(workingPointEntryType& workingPoint,
	    const std::vector<int>& categories, 
	    std::map<Int_t, TH1*>& histograms_mvaOutput_signal, std::map<Int_t, double>& categoryProbabilities_signal,
	    std::map<Int_t, TH1*>& histograms_mvaOutput_background, std::map<Int_t, double>& categoryProbabilities_background)
{
  std::cout << "<testWP>:" << std::endl;

  const double windowSize = 1.e-2; // needs to be multiple of bin-width

  double Ssum = 0.;
  double Bsum = 0.;
  for ( std::vector<int>::const_iterator category = categories.begin();
	category != categories.end(); ++category ) {
    std::cout << "category #"<< (*category) << ": cut = " << workingPoint.cuts_[*category] << std::endl;

    double p_signal = categoryProbabilities_signal[*category];
    double integral_signal = compIntegral(histograms_mvaOutput_signal[*category], workingPoint.cuts_[*category]);
    double dSdOutput = 
      TMath::Abs((compIntegral(histograms_mvaOutput_signal[*category], workingPoint.cuts_[*category]) 
                - compIntegral(histograms_mvaOutput_signal[*category], workingPoint.cuts_[*category] - windowSize))/windowSize);
    std::cout << " signal: integral = " << integral_signal << ", p = " << p_signal << " (p*dS/dOutput = "<< (p_signal*dSdOutput) << ")" << std::endl;
    Ssum += (p_signal*integral_signal);

    double p_background = categoryProbabilities_background[*category];
    double integral_background = compIntegral(histograms_mvaOutput_background[*category], workingPoint.cuts_[*category]);
    double dBdOutput = 
      TMath::Abs((compIntegral(histograms_mvaOutput_background[*category], workingPoint.cuts_[*category]) 
                - compIntegral(histograms_mvaOutput_background[*category], workingPoint.cuts_[*category] - windowSize))/windowSize);
    std::cout << " background: integral = " << integral_background << ", p = " << p_background << " (p*dB/dOutput = "<< (p_background*dBdOutput) << ")" << std::endl;
    Bsum += (p_background*integral_background);
  }

  std::cout << "--> S = "<< Ssum << ", B = " << Bsum << " --> S/B = " << (Ssum/Bsum) << std::endl;  
}

void writeWorkingPoints(const std::string& outputFileName, const std::string& outputTreeName, const std::vector<int>& categories, std::vector<workingPointEntryType>& workingPoints)
{
  TFile* outputFile = new TFile(outputFileName.data(), "RECREATE");
  TTree* outputTree = new TTree(outputTreeName.data(), outputTreeName.data());

  Float_t targetSignalEfficiency;
  outputTree->Branch("targetSignalEfficiency", &targetSignalEfficiency, "targetSignalEfficiency/F");

  Float_t minPt;
  outputTree->Branch("minPt", &minPt, "minPt/F");
  Float_t maxPt;
  outputTree->Branch("maxPt", &maxPt, "maxPt/F");

  std::map<Int_t, Float_t> cuts; // key = category
  for ( std::vector<int>::const_iterator category = categories.begin();
	category != categories.end(); ++category ) {
    std::string branchName = Form("cutCategory%i", *category);
    outputTree->Branch(branchName.data(), &cuts[*category], Form("%s/F", branchName.data()));
  }
    
  Float_t S;
  outputTree->Branch("S", &S, "S/F");
  Float_t B;
  outputTree->Branch("B", &B, "B/F");
  Float_t SoverB;
  outputTree->Branch("SoverB", &SoverB, "SoverB/F");
  
  for ( std::vector<workingPointEntryType>::iterator workingPoint = workingPoints.begin();
	workingPoint != workingPoints.end(); ++workingPoint ) {
    std::cout << "targetSignalEfficiency = " << workingPoint->targetSignalEfficiency_ << ":" << std::endl;
    targetSignalEfficiency = workingPoint->targetSignalEfficiency_;
    minPt = workingPoint->minPt_;
    maxPt = workingPoint->maxPt_;
    for ( std::vector<int>::const_iterator category = categories.begin();
	  category != categories.end(); ++category ) {
      std::cout << " category #" << (*category) << ": cut = " << workingPoint->cuts_[*category] << std::endl;
      cuts[*category] = workingPoint->cuts_[*category];
    }
    S = workingPoint->S_;
    B = workingPoint->B_;
    SoverB = ( workingPoint->B_ > 0. ) ? (workingPoint->S_/workingPoint->B_) : 1.e+6;
    std::cout << "S = " << workingPoint->S_ << ", B = " << workingPoint->B_ << " --> S/B = " << (workingPoint->S_/workingPoint->B_) << std::endl;
    std::cout << std::endl;
    outputTree->Fill();
  }

  std::cout << "output Tree:" << std::endl;
  //outputTree->Print();
  //outputTree->Scan("*", "", "", 20, 0);

  std::cout << "writing output Tree to file = " << outputFileName << "." << std::endl;
  outputFile->cd();
  outputTree->Write();

  delete outputFile;
}

typedef std::vector<std::string> vstring;
typedef std::vector<int> vint;
typedef std::vector<double> vdouble;

struct ptBinEntryType
{
  double minPt_;
  double maxPt_;
  std::map<Int_t, TH1*> histograms_mvaOutput_signal_;        // key = category
  std::map<Int_t, double> categoryProbabilities_signal_;     // key = category
  std::map<Int_t, TH1*> histograms_mvaOutput_background_;    // key = category
  std::map<Int_t, double> categoryProbabilities_background_; // key = category
};

int main(int argc, char* argv[]) 
{
//--- parse command-line arguments
  if ( argc < 2 ) {
    std::cout << "Usage: " << argv[0] << " [parameters.py]" << std::endl;
    return 0;
  }

  std::cout << "<computeWPcutsAntiElectronDiscrMVA>:" << std::endl;

//--- keep track of time it takes the macro to execute
  TBenchmark clock;
  clock.Start("computeWPcutsAntiElectronDiscrMVA");

//--- read python configuration parameters
  if ( !edm::readPSetsFrom(argv[1])->existsAs<edm::ParameterSet>("process") ) 
    throw cms::Exception("computeWPcutsAntiElectronDiscrMVA") 
      << "No ParameterSet 'process' found in configuration file = " << argv[1] << " !!\n";

  edm::ParameterSet cfg = edm::readPSetsFrom(argv[1])->getParameter<edm::ParameterSet>("process");

  edm::ParameterSet cfgComputeWPcutsAntiElectronDiscrMVA = cfg.getParameter<edm::ParameterSet>("computeWPcutsAntiElectronDiscrMVA");
  
  std::string inputTreeName = cfgComputeWPcutsAntiElectronDiscrMVA.getParameter<std::string>("inputTreeName");
  std::string outputTreeName = cfgComputeWPcutsAntiElectronDiscrMVA.getParameter<std::string>("outputTreeName");

  std::string branchName_mvaOutput = cfgComputeWPcutsAntiElectronDiscrMVA.getParameter<std::string>("branchName_mvaOutput");
  std::string branchName_categoryIdx = cfgComputeWPcutsAntiElectronDiscrMVA.getParameter<std::string>("branchName_categoryIdx");
  std::string branchName_tauPt = cfgComputeWPcutsAntiElectronDiscrMVA.getParameter<std::string>("branchName_tauPt");
  std::string branchName_logTauPt = cfgComputeWPcutsAntiElectronDiscrMVA.getParameter<std::string>("branchName_logTauPt");
  std::string branchName_evtWeight = cfgComputeWPcutsAntiElectronDiscrMVA.getParameter<std::string>("branchName_evtWeight");
  std::string branchName_classId = cfgComputeWPcutsAntiElectronDiscrMVA.getParameter<std::string>("branchName_classId");
  int classId_signal = cfgComputeWPcutsAntiElectronDiscrMVA.getParameter<int>("classId_signal");
  int classId_background = cfgComputeWPcutsAntiElectronDiscrMVA.getParameter<int>("classId_background");

  vint categories = cfgComputeWPcutsAntiElectronDiscrMVA.getParameter<vint>("categories");
  vdouble ptBinning = cfgComputeWPcutsAntiElectronDiscrMVA.getParameter<vdouble>("ptBinning");

  fwlite::InputSource inputFiles(cfg); 
  int maxEvents = inputFiles.maxEvents();
  std::cout << " maxEvents = " << maxEvents << std::endl;
  unsigned reportEvery = inputFiles.reportAfter();

  std::string outputFileName = cfgComputeWPcutsAntiElectronDiscrMVA.getParameter<std::string>("outputFileName");
  std::cout << " outputFileName = " << outputFileName << std::endl;

  TChain* inputTree = new TChain(inputTreeName.data());
  for ( vstring::const_iterator inputFileName = inputFiles.files().begin();
	inputFileName != inputFiles.files().end(); ++inputFileName ) {
    std::cout << "input Tree: adding file = " << (*inputFileName) << std::endl;
    inputTree->AddFile(inputFileName->data());
  }
  
  if ( !(inputTree->GetListOfFiles()->GetEntries() >= 1) ) {
    throw cms::Exception("computeWPcutsAntiElectronDiscrMVA") 
      << "Failed to identify input Tree !!\n";
  }

  std::cout << "input Tree contains " << inputTree->GetEntries() << " Entries in " << inputTree->GetListOfFiles()->GetEntries() << " files." << std::endl;

  // CV: need to call TChain::LoadTree before processing first event 
  //     in order to prevent ROOT causing a segmentation violation,
  //     cf. http://root.cern.ch/phpBB3/viewtopic.php?t=10062
  inputTree->LoadTree(0);

  Float_t mvaOutput;
  inputTree->SetBranchAddress(branchName_mvaOutput.data(), &mvaOutput);
  Float_t categoryIdx; // CV: TMVA stores Tau_Category branch in floating-point format !!
  inputTree->SetBranchAddress(branchName_categoryIdx.data(), &categoryIdx);
  Int_t classId;
  inputTree->SetBranchAddress(branchName_classId.data(), &classId);

  TTreeFormula* evtWeight_formula = 0;
  if ( branchName_evtWeight != "" ) {
    evtWeight_formula = new TTreeFormula("evtWeight_formula", branchName_evtWeight.data(), inputTree);   
  }

  int numPtBins = ptBinning.size() - 1;
  if ( !(numPtBins >= 2) ) {
    throw cms::Exception("computeWPcutsAntiElectronDiscrMVA") 
      << "Invalid Configuration Parameter 'ptBinning' !!\n";
  }
  std::vector<ptBinEntryType> ptBinEntries;
  for ( int iPtBin = 0; iPtBin < numPtBins; ++iPtBin ) {
    ptBinEntryType ptBinEntry;
    ptBinEntry.minPt_ = ptBinning[iPtBin];
    ptBinEntry.maxPt_ = ptBinning[iPtBin + 1];
    ptBinEntries.push_back(ptBinEntry);
  }
  
  enum { kLogTauPt, kTauPt };
  int mode = -1;
  Float_t logTauPt, tauPt;
  if ( branchName_logTauPt != "" && branchName_tauPt == "" ) {
    inputTree->SetBranchAddress(branchName_logTauPt.data(), &logTauPt);
    mode = kLogTauPt;
  } else if ( branchName_logTauPt == "" && branchName_tauPt != "" ) {
    inputTree->SetBranchAddress(branchName_tauPt.data(), &tauPt);
    mode = kTauPt;
  } 
  
  int currentTreeNumber = inputTree->GetTreeNumber();

  int numEntries = inputTree->GetEntries();
  for ( int iEntry = 0; iEntry < numEntries && (maxEvents == -1 || iEntry < maxEvents); ++iEntry ) {
    if ( iEntry > 0 && (iEntry % reportEvery) == 0 ) {
      std::cout << "processing Entry " << iEntry << std::endl;
    }
    
    inputTree->GetEntry(iEntry);

    if ( evtWeight_formula ) {
      // CV: need to call TTreeFormula::UpdateFormulaLeaves whenever input files changes in TChain
      //     in order to prevent ROOT causing a segmentation violation,
      //     cf. http://root.cern.ch/phpBB3/viewtopic.php?t=481
      if ( inputTree->GetTreeNumber() != currentTreeNumber ) {
	evtWeight_formula->UpdateFormulaLeaves();
	currentTreeNumber = inputTree->GetTreeNumber();
      }
    }
    
    double evtWeight = 1.0;
    if ( evtWeight_formula ) {
      evtWeight = evtWeight_formula->EvalInstance();
    }
    
    double tauPt_value;
    if      ( mode == kLogTauPt ) tauPt_value = TMath::Exp(logTauPt);
    else if ( mode == kTauPt    ) tauPt_value = tauPt;
    else assert(0);

    std::map<Int_t, TH1*>* histograms_mvaOutput = 0;
    std::map<Int_t, double>* categoryProbabilities = 0;
    std::string classId_string;
    const ptBinEntryType* ptBinEntry_ref = 0;
    for ( std::vector<ptBinEntryType>::iterator ptBinEntry = ptBinEntries.begin();
	  ptBinEntry != ptBinEntries.end(); ++ptBinEntry ) {
      if ( tauPt_value > ptBinEntry->minPt_ && tauPt_value < ptBinEntry->maxPt_ ) {
	if ( classId == classId_signal ) {
	  histograms_mvaOutput = &ptBinEntry->histograms_mvaOutput_signal_;
	  categoryProbabilities = &ptBinEntry->categoryProbabilities_signal_;
	  classId_string = "signal";
	  ptBinEntry_ref = &(*ptBinEntry);
	} else if ( classId == classId_background ) {
	  histograms_mvaOutput = &ptBinEntry->histograms_mvaOutput_background_;
	  categoryProbabilities = &ptBinEntry->categoryProbabilities_background_;
	  classId_string = "background";
	  ptBinEntry_ref = &(*ptBinEntry);
	} 
      }
    }

    //std::cout << "Entry #" << iEntry << ": classId = " << classId << ", categoryIdx = " << categoryIdx << ", mvaOutput = " << mvaOutput << ", evtWeight = " << evtWeight << std::endl;

    if ( histograms_mvaOutput && categoryProbabilities ) {
      for ( vint::const_iterator category = categories.begin();
	    category != categories.end(); ++category ) {
	if ( TMath::Nint(categoryIdx) & (1<<(*category)) ) {
	  TH1* histogram = (*histograms_mvaOutput)[*category];
	  if ( !histogram ) {
	    std::string histogramName = Form("histogram_mvaOutput_category%i_pt%1.0fto%1.0f_%s", *category, ptBinEntry_ref->minPt_, ptBinEntry_ref->maxPt_, classId_string.data());
	    histogram = new TH1D(histogramName.data(), histogramName.data(), 202000, -1.01, +1.01);
	    (*histograms_mvaOutput)[*category] = histogram;
	  }
	  histogram->Fill(mvaOutput, evtWeight);
	  (*categoryProbabilities)[*category] += evtWeight;
	}
      }
    } else {
      std::cerr << "Entry #" << iEntry << " has invalid classId = " << classId << " --> CHECK !!" << std::endl;
    }
  }

  std::cout << "--> " << inputTree->GetEntries() << " Entries processed." << std::endl;

  std::vector<workingPointEntryType> workingPoints;
  for ( std::vector<ptBinEntryType>::iterator ptBinEntry = ptBinEntries.begin();
	ptBinEntry != ptBinEntries.end(); ++ptBinEntry ) {
    std::cout << "Pt = " << ptBinEntry->minPt_ << ".." << ptBinEntry->maxPt_ << std::endl;
    
    // normalize categoryProbabilities
    double normalization_signal = 0.;
    double normalization_background = 0.;
    for ( vint::const_iterator category = categories.begin();
	  category != categories.end(); ++category ) {
      normalization_signal += ptBinEntry->categoryProbabilities_signal_[*category];
      normalization_background += ptBinEntry->categoryProbabilities_background_[*category];
    }
    for ( vint::const_iterator category = categories.begin();
	  category != categories.end(); ++category ) {
      ptBinEntry->categoryProbabilities_signal_[*category] /= normalization_signal;
      ptBinEntry->categoryProbabilities_background_[*category] /= normalization_background;
      std::cout << " category #" << (*category) << ":" 
		<< " P(signal) = " << ptBinEntry->categoryProbabilities_signal_[*category] << "," 
		<< " P(background) = " << ptBinEntry->categoryProbabilities_background_[*category] << std::endl;
    }

    // normalize histograms
    for ( vint::const_iterator category = categories.begin();
	  category != categories.end(); ++category ) {
      normalizeHistogram(ptBinEntry->histograms_mvaOutput_signal_[*category]);
      normalizeHistogram(ptBinEntry->histograms_mvaOutput_background_[*category]);
    }
  
    // compute working-points
    for ( double targetSignalEfficiency = 0.50; targetSignalEfficiency <= 0.995; targetSignalEfficiency += 0.01 ) {
      workingPointEntryType workingPoint = computeDifferentialWP(
        targetSignalEfficiency, 
        categories, 
        ptBinEntry->histograms_mvaOutput_signal_, ptBinEntry->categoryProbabilities_signal_, 
        ptBinEntry->histograms_mvaOutput_background_, ptBinEntry->categoryProbabilities_background_);

      testWP(workingPoint,
	     categories,
	     ptBinEntry->histograms_mvaOutput_signal_, ptBinEntry->categoryProbabilities_signal_,
	     ptBinEntry->histograms_mvaOutput_background_, ptBinEntry->categoryProbabilities_background_);

      workingPoint.minPt_ = ptBinEntry->minPt_;
      workingPoint.maxPt_ = ptBinEntry->maxPt_;

      workingPoints.push_back(workingPoint);    
    }
  }

  // write working-points to output file
  writeWorkingPoints(outputFileName, outputTreeName, categories, workingPoints);

  delete evtWeight_formula;
  delete inputTree;

  clock.Show("computeWPcutsAntiElectronDiscrMVA");

  return 0;
}
