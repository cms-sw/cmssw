
/** \executable preselectTreeTauIdMVA
 *
 * Preselect entries in TTree used for training MVA to identify hadronic tau decays.
 *
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: preselectTreeTauIdMVA.cc,v 1.1 2012/03/06 17:34:42 veelken Exp $
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

#include <iostream>
#include <string>
#include <vector>
#include <assert.h>

typedef std::vector<std::string> vstring;

int main(int argc, char* argv[]) 
{
//--- parse command-line arguments
  if ( argc < 2 ) {
    std::cout << "Usage: " << argv[0] << " [parameters.py]" << std::endl;
    return 0;
  }

  std::cout << "<preselectTreeTauIdMVA>:" << std::endl;

//--- keep track of time it takes the macro to execute
  TBenchmark clock;
  clock.Start("preselectTreeTauIdMVA");

//--- read python configuration parameters
  if ( !edm::readPSetsFrom(argv[1])->existsAs<edm::ParameterSet>("process") ) 
    throw cms::Exception("preselectTreeTauIdMVA") 
      << "No ParameterSet 'process' found in configuration file = " << argv[1] << " !!\n";

  edm::ParameterSet cfg = edm::readPSetsFrom(argv[1])->getParameter<edm::ParameterSet>("process");

  edm::ParameterSet cfgPreselectTreeTauIdMVA = cfg.getParameter<edm::ParameterSet>("preselectTreeTauIdMVA");
  
  std::string inputTreeName = cfgPreselectTreeTauIdMVA.getParameter<std::string>("inputTreeName");
  std::string outputTreeName = cfgPreselectTreeTauIdMVA.getParameter<std::string>("outputTreeName");

  vstring samples = cfgPreselectTreeTauIdMVA.getParameter<vstring>("samples");

  std::string preselection = cfgPreselectTreeTauIdMVA.getParameter<std::string>("preselection");

  std::string branchNamePt = cfgPreselectTreeTauIdMVA.getParameter<std::string>("branchNamePt");
  std::string branchNameEta = cfgPreselectTreeTauIdMVA.getParameter<std::string>("branchNameEta");
  std::string branchNameNumMatches = ( cfgPreselectTreeTauIdMVA.exists("branchNameNumMatches") ) ?
    cfgPreselectTreeTauIdMVA.getParameter<std::string>("branchNameNumMatches") : "";

  vstring inputVariables = cfgPreselectTreeTauIdMVA.getParameter<vstring>("inputVariables");

  vstring spectatorVariables = cfgPreselectTreeTauIdMVA.getParameter<vstring>("spectatorVariables");

  std::string branchNameEvtWeight = cfgPreselectTreeTauIdMVA.getParameter<std::string>("branchNameEvtWeight");

  bool keepAllBranches = cfgPreselectTreeTauIdMVA.getParameter<bool>("keepAllBranches");
  bool checkBranchesForNaNs = cfgPreselectTreeTauIdMVA.getParameter<bool>("checkBranchesForNaNs");

  int applyEventPruning = cfgPreselectTreeTauIdMVA.getParameter<int>("applyEventPruning");

  fwlite::InputSource inputFiles(cfg); 
  int maxEvents = inputFiles.maxEvents();
  std::cout << " maxEvents = " << maxEvents << std::endl;
  unsigned reportEvery = inputFiles.reportAfter();

  std::string outputFileName = cfgPreselectTreeTauIdMVA.getParameter<std::string>("outputFileName");
  std::cout << " outputFileName = " << outputFileName << std::endl;

  TChain* inputTree = new TChain(inputTreeName.data());
  for ( vstring::const_iterator inputFileName = inputFiles.files().begin();
	inputFileName != inputFiles.files().end(); ++inputFileName ) {
    bool matchesSample = false;
    for ( vstring::const_iterator sample = samples.begin();
	  sample != samples.end(); ++sample ) {
      if ( inputFileName->find(*sample) != std::string::npos ) matchesSample = true;
    }
    if ( matchesSample ) {
      std::cout << "input Tree: adding file = " << (*inputFileName) << std::endl;
      inputTree->AddFile(inputFileName->data());
    } 
  }
  
  if ( !(inputTree->GetListOfFiles()->GetEntries() >= 1) ) {
    throw cms::Exception("preselectTreeTauIdMVA") 
      << "Failed to identify input Tree !!\n";
  }

  // CV: need to call TChain::LoadTree before processing first event 
  //     in order to prevent ROOT causing a segmentation violation,
  //     cf. http://root.cern.ch/phpBB3/viewtopic.php?t=10062
  inputTree->LoadTree(0);

  vstring branchesToKeep_expressions = inputVariables;
  branchesToKeep_expressions.push_back(branchNameEvtWeight);
  branchesToKeep_expressions.push_back(branchNamePt);
  branchesToKeep_expressions.push_back(branchNameEta);
  if ( branchNameNumMatches != "" ) branchesToKeep_expressions.push_back(branchNameNumMatches);
  branchesToKeep_expressions.insert(branchesToKeep_expressions.end(), spectatorVariables.begin(), spectatorVariables.end());

  if ( keepAllBranches ) {
    TObjArray* branches = inputTree->GetListOfBranches();
    int numBranches = branches->GetEntries();
    for ( int iBranch = 0; iBranch < numBranches; ++iBranch ) {
      TBranch* branch = dynamic_cast<TBranch*>(branches->At(iBranch));
      assert(branch);
      std::string branchName = branch->GetName();
      branchesToKeep_expressions.push_back(branchName);
    }
  }

  std::cout << "input Tree contains " << inputTree->GetEntries() << " Entries in " << inputTree->GetListOfFiles()->GetEntries() << " files." << std::endl;
  std::cout << "preselecting Entries: preselection = '" << preselection << "'" << std::endl;
  TFile* outputFile = new TFile(outputFileName.data(), "RECREATE");
  TTree* outputTree = preselectTree(
    inputTree, outputTreeName, 
    preselection, branchesToKeep_expressions, 
    applyEventPruning, branchNamePt, branchNameEta, branchNameNumMatches,
    -1, false, false, 0, 0, 0,
    maxEvents, checkBranchesForNaNs, reportEvery);
  std::cout << "--> " << outputTree->GetEntries() << " Entries pass preselection." << std::endl;

  std::cout << "output Tree:" << std::endl;
  //outputTree->Print();
  //outputTree->Scan("*", "", "", 20, 0);

  std::cout << "writing output Tree to file = " << outputFileName << "." << std::endl;
  outputFile->cd();
  outputTree->Write();

  delete outputFile;

  delete inputTree;
  
  clock.Show("preselectTreeTauIdMVA");

  return 0;
}
