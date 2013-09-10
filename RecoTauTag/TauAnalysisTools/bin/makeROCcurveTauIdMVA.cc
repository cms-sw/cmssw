
/** \executable makeROCcurveTauIdMVA
 *
 * Fill histograms of MVA output for signal and background,
 * then make curves of backgreound rejection vs. signal efficiency
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

#include <TSystem.h>
#include <TFile.h>
#include <TChain.h>
#include <TTree.h>
#include <TTreeFormula.h>
#include <TString.h>
#include <TBenchmark.h>
#include <TH1.h>
#include <TGraph.h>
#include <TROOT.h>

#include <iostream>
#include <string>
#include <vector>
#include <assert.h>

typedef std::vector<std::string> vstring;

namespace
{
  struct plotEntryType
  {
    plotEntryType(const std::string& histogramName, int numBins, double min, double max,double minTauPt, double maxTauPt)
      : numBins_(numBins),
	min_(min),
	max_(max),
	minTauPt_(minTauPt),
	maxTauPt_(maxTauPt),
	histogram_(0)
    {
      histogramName_ = histogramName;
      histogramName_.append(getTauPtLabel(minTauPt_, maxTauPt_));
    }
    ~plotEntryType() 
    {
      delete histogram_;
    }
    void bookHistograms()
    {
      std::cout << "booking histogram = " << histogramName_ << std::endl;
      histogram_ = new TH1D(histogramName_.data(), histogramName_.data(), numBins_, min_, max_);
    }
    void fillHistograms(double tauPt, double discriminator, double evtWeight)
    {
      if ( (minTauPt_ < 0. || tauPt > minTauPt_) && 
	   (maxTauPt_ < 0. || tauPt < maxTauPt_) ) {
	histogram_->Fill(discriminator, evtWeight);
      }
    }
    int numBins_;
    double min_;
    double max_;
    double minTauPt_;
    double maxTauPt_;
    std::string tauPtLabel_;
    std::string histogramName_;
    TH1* histogram_;
  };

  void fillHistograms(TTree* tree, const std::string& disriminator, 
		      const std::string& branchNameLogTauPt, const std::string& branchNameTauPt, const std::string& branchNameEvtWeight, 
		      std::vector<plotEntryType*>& plotEntries,
		      unsigned reportEvery)
  {
    std::cout << "<fillHistograms>:" << std::endl;
    std::string treeFormulaName = Form("%s_formula", tree->GetName());
    TTreeFormula* treeFormula = new TTreeFormula(treeFormulaName.data(), disriminator.data(), tree);    
    std::cout << " treeFormula = " << disriminator.data() << std::endl;

    //std::cout << "tree:" << std::endl;
    //tree->Print();

    enum { kLogTauPt, kTauPt };
    int mode = -1;
    Float_t logTauPt, tauPt;
    if ( branchNameLogTauPt != "" && branchNameTauPt == "" ) {
      tree->SetBranchAddress(branchNameLogTauPt.data(), &logTauPt);
      mode = kLogTauPt;
    } else if ( branchNameLogTauPt == "" && branchNameTauPt != "" ) {
      tree->SetBranchAddress(branchNameTauPt.data(), &tauPt);
      mode = kTauPt;
    } 

    Float_t evtWeight = 1.0;
    if ( branchNameEvtWeight != "" ) {
      tree->SetBranchAddress(branchNameEvtWeight.data(), &evtWeight);
    }

    int currentTreeNumber = tree->GetTreeNumber();

    int numEntries = tree->GetEntries();
    for ( int iEntry = 0; iEntry < numEntries; ++iEntry ) {
      if ( iEntry > 0 && (iEntry % reportEvery) == 0 ) {
	std::cout << "processing Entry " << iEntry << std::endl;
      }
    
      tree->GetEntry(iEntry);

      double tauPt_value;
      if      ( mode == kLogTauPt ) tauPt_value = TMath::Exp(logTauPt);
      else if ( mode == kTauPt    ) tauPt_value = tauPt;
      else assert(0);

      // CV: need to call TTreeFormula::UpdateFormulaLeaves whenever input files changes in TChain
      //     in order to prevent ROOT causing a segmentation violation,
      //     cf. http://root.cern.ch/phpBB3/viewtopic.php?t=481
      if ( tree->GetTreeNumber() != currentTreeNumber ) {
	treeFormula->UpdateFormulaLeaves();
	currentTreeNumber = tree->GetTreeNumber();
      }

      double discriminator = treeFormula->EvalInstance();

      for ( std::vector<plotEntryType*>::iterator plotEntry = plotEntries.begin();
	    plotEntry != plotEntries.end(); ++plotEntry ) {
	(*plotEntry)->fillHistograms(tauPt_value, discriminator, evtWeight);
      }
    }

    delete treeFormula;
  }

  void normalizeHistograms(std::vector<plotEntryType*>& plotEntries)
  {
    for ( std::vector<plotEntryType*>::iterator plotEntry = plotEntries.begin();
	  plotEntry != plotEntries.end(); ++plotEntry ) {      
      TH1* histogram = (*plotEntry)->histogram_;
      if ( histogram->Integral() > 0. ) {
	if ( !histogram->GetSumw2N() ) histogram->Sumw2();
	histogram->Scale(1./histogram->Integral());
      } 
    }
  }

  TGraph* makeROCcurve(const std::string& graphName, const TH1* histogram_signal, const TH1* histogram_background)
  {
    assert(histogram_signal->GetNbinsX() == histogram_background->GetNbinsX());
    int numBins = histogram_signal->GetNbinsX();    

    std::vector<double> graphPointsX;
    std::vector<double> graphPointsY;

    double mean_signal = histogram_signal->GetMean();
    double mean_background = histogram_background->GetMean();
    std::cout << "mean: signal = " << mean_signal << ", background = " << mean_background << std::endl;
    int initialBin, increment;
    if ( mean_signal > mean_background ) { // integrate from "left"
      initialBin = 0; // underflow bin
      increment = +1;
    } else { // integrate from "right"
      initialBin = histogram_signal->GetNbinsX() + 1; // overflow bin
      increment = -1;
    }

    double runningSum_signal = 0.;
    double normalization_signal = histogram_signal->Integral();
    double runningSum_background = 0.;
    double normalization_background = histogram_background->Integral();
    for ( int iBin = initialBin; iBin >= 0 && iBin <= (numBins + 1); iBin += increment ) {
      assert(histogram_signal->GetBinCenter(iBin) == histogram_background->GetBinCenter(iBin));
      runningSum_signal += histogram_signal->GetBinContent(iBin);
      runningSum_background += histogram_background->GetBinContent(iBin);
      std::cout << "running sum(bin #" << iBin << "): signal = " << runningSum_signal << ", background = " << runningSum_background << std::endl;
      if ( (runningSum_signal >= 1.e-6 || runningSum_background >= 1.e-6) &&
	   ((normalization_signal - runningSum_signal) >= 1.e-6 || (normalization_background - runningSum_background) >= 1.e-6) ) {
	double x = 1. - (runningSum_signal/normalization_signal);
	graphPointsX.push_back(x);
	double y = runningSum_background/normalization_background;
	graphPointsY.push_back(y);
      }
    }
    assert(graphPointsX.size() == graphPointsY.size());

    TGraph* graph = 0;

    int numPoints = graphPointsX.size();
    if ( numPoints >= 1 ) {
      graph = new TGraph(numPoints);
      for ( int iPoint = 0; iPoint < numPoints; ++iPoint ) {
	graph->SetPoint(iPoint, graphPointsX[iPoint], graphPointsY[iPoint]);
      }
    } else {
      std::cerr << "Warning: failed to compute efficiency and fake-rate for graph = " << graphName << " !!" << std::endl;
      graph = new TGraph(1);
      graph->SetPoint(0, 0.5, 0.5);
    }
    graph->SetName(graphName.data());
    std::cout << "graph = " << graph->GetName() << ": #points = " << graph->GetN() << std::endl;

    return graph;
  }
}

int main(int argc, char* argv[]) 
{
//--- parse command-line arguments
  if ( argc < 2 ) {
    std::cout << "Usage: " << argv[0] << " [parameters.py]" << std::endl;
    return 0;
  }

  std::cout << "<makeROCcurveTauIdMVA>:" << std::endl;

//--- keep track of time it takes the macro to execute
  TBenchmark clock;
  clock.Start("makeROCcurveTauIdMVA");

//--- CV: disable automatic association of histograms to files
//       (default behaviour ROOT object ownership, 
//        cf. http://root.cern.ch/download/doc/Users_Guide_5_26.pdf section 8)
  TH1::AddDirectory(false);

//--- read python configuration parameters
  if ( !edm::readPSetsFrom(argv[1])->existsAs<edm::ParameterSet>("process") ) 
    throw cms::Exception("makeROCcurveTauIdMVA") 
      << "No ParameterSet 'process' found in configuration file = " << argv[1] << " !!\n";

  edm::ParameterSet cfg = edm::readPSetsFrom(argv[1])->getParameter<edm::ParameterSet>("process");

  edm::ParameterSet cfgMakeROCcurveTauIdMVA = cfg.getParameter<edm::ParameterSet>("makeROCcurveTauIdMVA");
  
  std::string treeName = cfgMakeROCcurveTauIdMVA.getParameter<std::string>("treeName");

  std::string preselection_signal;
  if ( cfgMakeROCcurveTauIdMVA.exists("preselection_signal") ) {
    preselection_signal = cfgMakeROCcurveTauIdMVA.getParameter<std::string>("preselection_signal");
  } else {
    preselection_signal = cfgMakeROCcurveTauIdMVA.getParameter<std::string>("preselection");
  }
  if ( preselection_signal.length() >= 1 ) {
    preselection_signal = Form("(%s)", preselection_signal.data());
  }
  std::string preselection_background;
  if ( cfgMakeROCcurveTauIdMVA.exists("preselection_background") ) {
    preselection_background = cfgMakeROCcurveTauIdMVA.getParameter<std::string>("preselection_background");
  } else {
    preselection_background = cfgMakeROCcurveTauIdMVA.getParameter<std::string>("preselection");
  }
  if ( preselection_background.length() >= 1 ) {
    preselection_background = Form("(%s)", preselection_background.data());
  }

  int numMethods_signal_and_background_separation = 0;
  vstring signalSamples;
  vstring backgroundSamples;
  if ( cfgMakeROCcurveTauIdMVA.exists("signalSamples") and cfgMakeROCcurveTauIdMVA.exists("backgroundSamples") ) {
    signalSamples = cfgMakeROCcurveTauIdMVA.getParameter<vstring>("signalSamples");
    backgroundSamples = cfgMakeROCcurveTauIdMVA.getParameter<vstring>("backgroundSamples");
    ++numMethods_signal_and_background_separation;
  } 
  std::string branchNameClassId;
  if ( cfgMakeROCcurveTauIdMVA.exists("classId_signal") && cfgMakeROCcurveTauIdMVA.exists("classId_background") ) {
    branchNameClassId = cfgMakeROCcurveTauIdMVA.getParameter<std::string>("branchNameClassId");
    int classId_signal = cfgMakeROCcurveTauIdMVA.getParameter<int>("classId_signal");
    int classId_background = cfgMakeROCcurveTauIdMVA.getParameter<int>("classId_background");
    if ( preselection_signal.length() > 0 ) preselection_signal.append(" && ");
    preselection_signal.append(Form("%s == %i", branchNameClassId.data(), classId_signal));
    if ( preselection_background.length() > 0 ) preselection_background.append(" && ");
    preselection_background.append(Form("%s == %i", branchNameClassId.data(), classId_background));
    ++numMethods_signal_and_background_separation;
  }
  if ( numMethods_signal_and_background_separation != 1 ) 
    throw cms::Exception("makeROCcurveTauIdMVA") 
      << "Need to specify either Configuration parameters 'signalSamples' and 'backgroundSamples' or 'classId_signal' plus 'classId_background' !!\n";

  std::string discriminator = cfgMakeROCcurveTauIdMVA.getParameter<std::string>("discriminator");

  std::string branchNameLogTauPt = cfgMakeROCcurveTauIdMVA.getParameter<std::string>("branchNameLogTauPt");
  std::string branchNameTauPt = cfgMakeROCcurveTauIdMVA.getParameter<std::string>("branchNameTauPt");
  if ( (branchNameLogTauPt == "" && branchNameTauPt == "") ||
       (branchNameLogTauPt != "" && branchNameTauPt != "") )
    throw cms::Exception("makeROCcurveTauIdMVA") 
      << "Need to set either Configuration parameters 'branchNameLogTauPt' or 'branchNameTauPt' to non-zero values !!\n";

  std::string branchNameEvtWeight = cfgMakeROCcurveTauIdMVA.getParameter<std::string>("branchNameEvtWeight");

  std::string graphName = cfgMakeROCcurveTauIdMVA.getParameter<std::string>("graphName");
  std::string histogramName = Form("%s_histogram", graphName.data());
  edm::ParameterSet cfgBinning = cfgMakeROCcurveTauIdMVA.getParameter<edm::ParameterSet>("binning");
  int numBins = cfgBinning.getParameter<int>("numBins");
  double min = cfgBinning.getParameter<double>("min");
  double max = cfgBinning.getParameter<double>("max");

  fwlite::InputSource inputFiles(cfg); 
  int maxEvents = inputFiles.maxEvents();
  unsigned reportEvery = inputFiles.reportAfter();

  TChain* tree_signal = new TChain(treeName.data());
  TChain* tree_background = new TChain(treeName.data());
  for ( vstring::const_iterator inputFileName = inputFiles.files().begin();
	inputFileName != inputFiles.files().end(); ++inputFileName ) {
    if ( signalSamples.size() > 0 && backgroundSamples.size() > 0 ) {
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
      if ( matchesSample_signal && matchesSample_background ) {
	throw cms::Exception("trainTauIdMVA") 
	  << "Failed to identify if inputFile = " << (*inputFileName) << " is signal or background !!\n";
      }
      if ( matchesSample_signal ) {
	std::cout << "signal Tree: adding file = " << (*inputFileName) << std::endl;	
	tree_signal->AddFile(inputFileName->data());
      } 
      if ( matchesSample_background ) {
	std::cout << "background Tree: adding file = " << (*inputFileName) << std::endl;
	tree_background->AddFile(inputFileName->data());
      }
    } else {
      tree_signal->AddFile(inputFileName->data());
      tree_background = tree_signal;
    }
  }
  
  if ( !(tree_signal->GetListOfFiles()->GetEntries() >= 1) ) {
    throw cms::Exception("trainTauIdMVA") 
      << "Failed to identify signal Tree !!\n";
  }
  if ( !(tree_background->GetListOfFiles()->GetEntries() >= 1) ) {
    throw cms::Exception("trainTauIdMVA") 
      << "Failed to identify background Tree !!\n";
  }

  std::vector<TTree*> treesToDelete;
  treesToDelete.push_back(tree_signal);
  if ( tree_background != tree_signal ) treesToDelete.push_back(tree_background);
  
  // CV: need to call TChain::LoadTree before processing first event 
  //     in order to prevent ROOT causing a segmentation violation,
  //     cf. http://root.cern.ch/phpBB3/viewtopic.php?t=10062
  tree_signal->LoadTree(0);
  if ( tree_background != tree_signal ) tree_background->LoadTree(0);

  vstring branchesToKeep_expressions;
  branchesToKeep_expressions.push_back(branchNameLogTauPt.data());
  branchesToKeep_expressions.push_back(branchNameTauPt.data());  
  branchesToKeep_expressions.push_back(discriminator);
  branchesToKeep_expressions.push_back(branchNameEvtWeight);

  typedef std::pair<double, double> pdouble;
  std::vector<pdouble> tauPtBins;
  tauPtBins.push_back(pdouble(  -1.,   50.));
  tauPtBins.push_back(pdouble(  50.,  100.));
  tauPtBins.push_back(pdouble( 100.,  200.));
  tauPtBins.push_back(pdouble( 200.,  400.));
  tauPtBins.push_back(pdouble( 400.,  600.));
  tauPtBins.push_back(pdouble( 600.,  900.));
  tauPtBins.push_back(pdouble( 900., 1200.));
  tauPtBins.push_back(pdouble(1200.,   -1.));
  tauPtBins.push_back(pdouble(  -1.,   -1.));
  
  std::string outputFileName = cfgMakeROCcurveTauIdMVA.getParameter<std::string>("outputFileName");
  std::cout << " outputFileName = " << outputFileName << std::endl;
  TFile* outputFile = new TFile(outputFileName.data(), "RECREATE");

  std::cout << "signal Tree contains " << tree_signal->GetEntries() << " Entries in " << tree_signal->GetListOfFiles()->GetEntries() << " files." << std::endl;
  std::cout << "preselecting Entries: preselection = '" << preselection_signal << "'" << std::endl;
  TTree* preselectedTree_signal = preselectTree(
    tree_signal, "preselectedTree_signal", 
    preselection_signal, branchesToKeep_expressions, 
    false, "", "", "", 
    -1, false, false, 0, 0, 0,
    maxEvents, true, reportEvery);
  std::cout << "--> " << preselectedTree_signal->GetEntries() << " signal Entries pass preselection." << std::endl;
  std::vector<plotEntryType*> plotEntries_signal;
  for ( std::vector<pdouble>::const_iterator tauPtBin = tauPtBins.begin();
	tauPtBin != tauPtBins.end(); ++tauPtBin ) {
    std::cout << "creating signal histograms for tauPtBin = " << tauPtBin->first << ".." << tauPtBin->second << std::endl;
    plotEntryType* plotEntry_signal = new plotEntryType(Form("%s_signal", histogramName.data()), numBins, min, max, tauPtBin->first, tauPtBin->second);
    plotEntry_signal->bookHistograms();
    plotEntries_signal.push_back(plotEntry_signal);
  }
  fillHistograms(
    preselectedTree_signal, 
    discriminator, branchNameLogTauPt, branchNameTauPt, branchNameEvtWeight, 
    plotEntries_signal, 
    reportEvery);
  delete preselectedTree_signal;

  std::cout << "background Tree contains " << tree_background->GetEntries() << " Entries in " << tree_background->GetListOfFiles()->GetEntries() << " files." << std::endl;
  std::cout << "preselecting Entries: preselection = '" << preselection_background << "'" << std::endl;
  TTree* preselectedTree_background = preselectTree(
    tree_background, "preselectedTree_background", 
    preselection_background, branchesToKeep_expressions, 
    false, "", "", "", 
    -1, false, false, 0, 0, 0,
    maxEvents, true, reportEvery);
  std::cout << "--> " << preselectedTree_background->GetEntries() << " background Entries pass preselection." << std::endl;
  std::vector<plotEntryType*> plotEntries_background;
  for ( std::vector<pdouble>::const_iterator tauPtBin = tauPtBins.begin();
	tauPtBin != tauPtBins.end(); ++tauPtBin ) {
    std::cout << "creating background histograms for tauPtBin = " << tauPtBin->first << ".." << tauPtBin->second << std::endl;
    plotEntryType* plotEntry_background = new plotEntryType(Form("%s_background", histogramName.data()), numBins, min, max, tauPtBin->first, tauPtBin->second);
    plotEntry_background->bookHistograms();
    plotEntries_background.push_back(plotEntry_background);
  }
  fillHistograms(
    preselectedTree_background, 
    discriminator, branchNameLogTauPt, branchNameTauPt, branchNameEvtWeight, 
    plotEntries_background,
    reportEvery);
  delete preselectedTree_background;

  for ( std::vector<TTree*>::iterator it = treesToDelete.begin();
	it != treesToDelete.end(); ++it ) {
    delete (*it);
  }

  normalizeHistograms(plotEntries_signal);
  normalizeHistograms(plotEntries_background);
    
  std::vector<TGraph*> graphsROCcurve;
  assert(plotEntries_signal.size() == plotEntries_background.size());
  size_t numTauPtBins = plotEntries_signal.size();
  for ( size_t iTauPtBin = 0; iTauPtBin < numTauPtBins; ++iTauPtBin ) {
    std::cout << "computing ROC curve for tauPtBin = " << tauPtBins[iTauPtBin].first << ".." << tauPtBins[iTauPtBin].second << std::endl;
    const TH1* histogram_signal = plotEntries_signal[iTauPtBin]->histogram_;
    const TH1* histogram_background = plotEntries_background[iTauPtBin]->histogram_;
    TGraph* graphROCcurve = makeROCcurve(Form("%s%s", graphName.data(), getTauPtLabel(tauPtBins[iTauPtBin].first, tauPtBins[iTauPtBin].second).data()), histogram_signal, histogram_background);
    graphsROCcurve.push_back(graphROCcurve);
  }

  outputFile->cd();
  for ( std::vector<plotEntryType*>::iterator plotEntry = plotEntries_signal.begin();
	plotEntry != plotEntries_signal.end(); ++plotEntry ) {
    std::cout << "writing histogram = " << (*plotEntry)->histogram_->GetName() << std::endl;
    (*plotEntry)->histogram_->Write();
  }
  for ( std::vector<plotEntryType*>::iterator plotEntry = plotEntries_background.begin();
	plotEntry != plotEntries_background.end(); ++plotEntry ) {
    std::cout << "writing histogram = " << (*plotEntry)->histogram_->GetName() << std::endl;
    (*plotEntry)->histogram_->Write();
  }
  for ( std::vector<TGraph*>::iterator graphROCcurve = graphsROCcurve.begin();
	graphROCcurve != graphsROCcurve.end(); ++graphROCcurve ) {    
    std::cout << "writing graph = " << (*graphROCcurve)->GetName() << std::endl;
    (*graphROCcurve)->Write();
  }
  delete outputFile;

  for ( std::vector<plotEntryType*>::iterator it = plotEntries_signal.begin();
	it != plotEntries_signal.end(); ++it ) {
    std::cout << "deleting histogram = " << (*it)->histogram_->GetName() << std::endl;
    delete (*it);
  }
  for ( std::vector<plotEntryType*>::iterator it = plotEntries_background.begin();
	it != plotEntries_background.end(); ++it ) {
    std::cout << "deleting histogram = " << (*it)->histogram_->GetName() << std::endl;
    delete (*it);
  }
  for ( std::vector<TGraph*>::iterator it = graphsROCcurve.begin();
	it != graphsROCcurve.end(); ++it ) { 
    std::cout << "deleting graph = " << (*it)->GetName() << std::endl;
    delete (*it);
  }

  clock.Show("makeROCcurveTauIdMVA");

  return 0;
}
