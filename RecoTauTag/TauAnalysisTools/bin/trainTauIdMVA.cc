
/** \executable trainTauIdMVA
 *
 * Train MVA for identifying hadronic tau decays
 *
 * NOTE: The MVA need to be of type BDTG so that it can be stored as object of type
 *         CondFormats/EgammaObjects/interface/GBRForest.h
 *       in a ROOT file, an SQLlite file or in the DataBase
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

#include "CondFormats/EgammaObjects/interface/GBRForest.h"

#include "RecoTauTag/TauAnalysisTools/bin/tauIdMVATrainingAuxFunctions.h"

#include "TMVA/Factory.h"
#include "TMVA/MethodBase.h"
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"

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

int main(int argc, char* argv[]) 
{
//--- parse command-line arguments
  if ( argc < 2 ) {
    std::cout << "Usage: " << argv[0] << " [parameters.py]" << std::endl;
    return 0;
  }

  std::cout << "<trainTauIdMVA>:" << std::endl;

//--- keep track of time it takes the macro to execute
  TBenchmark clock;
  clock.Start("trainTauIdMVA");

//--- read python configuration parameters
  if ( !edm::readPSetsFrom(argv[1])->existsAs<edm::ParameterSet>("process") ) 
    throw cms::Exception("trainTauIdMVA") 
      << "No ParameterSet 'process' found in configuration file = " << argv[1] << " !!\n";

  edm::ParameterSet cfg = edm::readPSetsFrom(argv[1])->getParameter<edm::ParameterSet>("process");

  edm::ParameterSet cfgTrainTauIdMVA = cfg.getParameter<edm::ParameterSet>("trainTauIdMVA");
  
  std::string treeName = cfgTrainTauIdMVA.getParameter<std::string>("treeName");

  vstring signalSamples = cfgTrainTauIdMVA.getParameter<vstring>("signalSamples");
  vstring backgroundSamples = cfgTrainTauIdMVA.getParameter<vstring>("backgroundSamples");

  bool applyPtReweighting = cfgTrainTauIdMVA.getParameter<bool>("applyPtReweighting");
  bool applyEtaReweighting = cfgTrainTauIdMVA.getParameter<bool>("applyEtaReweighting");
  TString reweightOption_tstring = cfgTrainTauIdMVA.getParameter<std::string>("reweight").data();  
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
    else throw cms::Exception("trainTauIdMVA") 
      << "Invalid Configuration parameter 'reweight' = " << reweightOption_tstring.Data() << " !!\n";
  }

  vstring inputVariables = cfgTrainTauIdMVA.getParameter<vstring>("inputVariables");

  vstring spectatorVariables = cfgTrainTauIdMVA.getParameter<vstring>("spectatorVariables");

  std::string branchNameEvtWeight = cfgTrainTauIdMVA.getParameter<std::string>("branchNameEvtWeight");

  fwlite::InputSource inputFiles(cfg); 

  std::string outputFileName = cfgTrainTauIdMVA.getParameter<std::string>("outputFileName");
  std::cout << " outputFileName = " << outputFileName << std::endl;
  TFile* outputFile = new TFile(outputFileName.data(), "RECREATE");

  TChain* tree_signal = new TChain(treeName.data());
  TChain* tree_background = new TChain(treeName.data());
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
  }
  
  if ( !(tree_signal->GetListOfFiles()->GetEntries() >= 1) ) {
    throw cms::Exception("trainTauIdMVA") 
      << "Failed to identify signal Tree !!\n";
  }
  if ( !(tree_background->GetListOfFiles()->GetEntries() >= 1) ) {
    throw cms::Exception("trainTauIdMVA") 
      << "Failed to identify background Tree !!\n";
  }

  // CV: need to call TChain::LoadTree before processing first event 
  //     in order to prevent ROOT causing a segmentation violation,
  //     cf. http://root.cern.ch/phpBB3/viewtopic.php?t=10062
  //tree_signal->LoadTree(0);
  //tree_background->LoadTree(0);

  std::cout << "signal Tree contains " << tree_signal->GetEntries() << " Entries in " << tree_signal->GetListOfFiles()->GetEntries() << " files." << std::endl;
  tree_signal->Print();
  tree_signal->Scan("*", "", "", 20, 0);

  std::cout << "background Tree contains " << tree_background->GetEntries() << " Entries in " << tree_background->GetListOfFiles()->GetEntries() << " files." << std::endl;
  tree_background->Print();
  tree_background->Scan("*", "", "", 20, 0);

//--- train MVA
  std::string mvaName = cfgTrainTauIdMVA.getParameter<std::string>("mvaName");
  std::string mvaMethodType = cfgTrainTauIdMVA.getParameter<std::string>("mvaMethodType");
  std::string mvaMethodName = cfgTrainTauIdMVA.getParameter<std::string>("mvaMethodName");

  std::string mvaTrainingOptions = cfgTrainTauIdMVA.getParameter<std::string>("mvaTrainingOptions");

  TMVA::Tools::Instance();
  TMVA::Factory* factory = new TMVA::Factory(mvaName.data(), outputFile, "!V:!Silent");
  
  factory->AddSignalTree(tree_signal);
  factory->AddBackgroundTree(tree_background);

  for ( vstring::const_iterator inputVariable = inputVariables.begin();
	inputVariable != inputVariables.end(); ++inputVariable ) {
    int idx = inputVariable->find_last_of("/");
    if ( idx == (int(inputVariable->length()) - 2) ) {
      std::string inputVariableName = std::string(*inputVariable, 0, idx);      
      char inputVariableType = (*inputVariable)[idx + 1];
      factory->AddVariable(inputVariableName.data(), inputVariableType);
    } else {
      throw cms::Exception("trainTauIdMVA") 
	<< "Failed to determine name & type for inputVariable = " << (*inputVariable) << " !!\n";
    }
  }
  for ( vstring::const_iterator spectatorVariable = spectatorVariables.begin();
	spectatorVariable != spectatorVariables.end(); ++spectatorVariable ) {
    int idxSpectatorVariable = spectatorVariable->find_last_of("/");
    std::string spectatorVariableName = std::string(*spectatorVariable, 0, idxSpectatorVariable);      
    bool isInputVariable = false;
    for ( vstring::const_iterator inputVariable = inputVariables.begin();
	  inputVariable != inputVariables.end(); ++inputVariable ) {
      int idxInputVariable = inputVariable->find_last_of("/");
      std::string inputVariableName = std::string(*inputVariable, 0, idxInputVariable); 
      if ( spectatorVariableName == inputVariableName ) isInputVariable = true;
    }
    if ( !isInputVariable ) {
      factory->AddSpectator(spectatorVariableName.data());
    }
  }
  if ( (applyPtReweighting || applyEtaReweighting) && reweight_or_KILL == kReweight && 
       (reweightOption == kReweight_or_KILLsignal || reweightOption == kReweight_or_KILLflat || reweightOption == kReweight_or_KILLmin) ) {
    std::string signalWeightExpression = "ptVsEtaReweight";
    if ( branchNameEvtWeight != "" ) signalWeightExpression.append("*").append(branchNameEvtWeight);
    factory->SetSignalWeightExpression(signalWeightExpression.data());
  } else {
    if ( branchNameEvtWeight != "" ) factory->SetSignalWeightExpression(branchNameEvtWeight.data());
  }
  if ( (applyPtReweighting || applyEtaReweighting) && reweight_or_KILL == kReweight && 
       (reweightOption == kReweight_or_KILLbackground || reweightOption == kReweight_or_KILLflat || reweightOption == kReweight_or_KILLmin) ) {
    std::string backgroundWeightExpression = "ptVsEtaReweight";
    if ( branchNameEvtWeight != "" ) backgroundWeightExpression.append("*").append(branchNameEvtWeight);
    factory->SetBackgroundWeightExpression(backgroundWeightExpression.data());
  } else {
    if ( branchNameEvtWeight != "" ) factory->SetBackgroundWeightExpression(branchNameEvtWeight.data());
  }

  TCut cut = "";
  factory->PrepareTrainingAndTestTree(cut, "nTrain_Signal=0:nTrain_Background=0:nTest_Signal=0:nTest_Background=0:SplitMode=Random:NormMode=NumEvents:!V");
  factory->BookMethod(mvaMethodType.data(), mvaMethodName.data(), mvaTrainingOptions.data());

  std::cout << "Info: calling TMVA::Factory::TrainAllMethods" << std::endl;
  factory->TrainAllMethods();
  std::cout << "Info: calling TMVA::Factory::TestAllMethods" << std::endl;
  factory->TestAllMethods();
  std::cout << "Info: calling TMVA::Factory::EvaluateAllMethods" << std::endl;
  factory->EvaluateAllMethods();  
  
  delete factory;
  TMVA::Tools::DestroyInstance();
  delete outputFile;

  delete tree_signal;
  delete tree_background;

  std::cout << "Info: converting MVA to GBRForest format" << std::endl;
  TMVA::Tools::Instance();
  TMVA::Reader* reader = new TMVA::Reader("!V:!Silent");
  Float_t dummyVariable;
  for ( vstring::const_iterator inputVariable = inputVariables.begin();
	inputVariable != inputVariables.end(); ++inputVariable ) {
    int idx = inputVariable->find_last_of("/");
    std::string inputVariableName = std::string(*inputVariable, 0, idx);
    reader->AddVariable(inputVariableName.data(), &dummyVariable);    
  }
  for ( vstring::const_iterator spectatorVariable = spectatorVariables.begin();
	spectatorVariable != spectatorVariables.end(); ++spectatorVariable ) {
    int idxSpectatorVariable = spectatorVariable->find_last_of("/");
    std::string spectatorVariableName = std::string(*spectatorVariable, 0, idxSpectatorVariable);      
    bool isInputVariable = false;
    for ( vstring::const_iterator inputVariable = inputVariables.begin();
	  inputVariable != inputVariables.end(); ++inputVariable ) {
      int idxInputVariable = inputVariable->find_last_of("/");
      std::string inputVariableName = std::string(*inputVariable, 0, idxInputVariable); 
      if ( spectatorVariableName == inputVariableName ) isInputVariable = true;
    }
    if ( !isInputVariable ) {
      reader->AddSpectator(spectatorVariableName.data(), &dummyVariable);
    }
  }
  TMVA::IMethod* mva = reader->BookMVA(mvaMethodName.data(), Form("weights/%s_%s.weights.xml", mvaName.data(), mvaMethodName.data()));
  saveAsGBRForest(mva, mvaName, outputFileName);
  delete mva;
  
  clock.Show("trainTauIdMVA");

  return 0;
}
