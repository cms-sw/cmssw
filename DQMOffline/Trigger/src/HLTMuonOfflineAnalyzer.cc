// -*- C++ -*-
//
// Package:     HLTMuonOfflineAnalyzer
// Class:       HLTMuonOfflineAnalyzer
// 

//
// Jason Slaunwhite and Jeff Klukas
//
//

// system include files
#include <memory>
#include <iostream>

// user include files
#include "DQMOffline/Trigger/interface/HLTMuonMatchAndPlotContainer.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


#include "TFile.h"
#include "TDirectory.h"
#include "TPRegexp.h"


//////////////////////////////////////////////////////////////////////////////
//////// Define the interface ////////////////////////////////////////////////



class HLTMuonOfflineAnalyzer : public edm::EDAnalyzer {

public:

  explicit HLTMuonOfflineAnalyzer(const edm::ParameterSet&);

private:

  // Analyzer Methods
  virtual void beginJob() override;
  virtual void beginRun(const edm::Run &, const edm::EventSetup &) override;
  virtual void analyze(const edm::Event &, const edm::EventSetup &) override;
  virtual void endRun(const edm::Run &, const edm::EventSetup &) override;
  virtual void endJob() override;

  // Extra Methods
  std::vector<std::string> moduleLabels(std::string);

  // Input from Configuration File
  edm::ParameterSet pset_;
  std::string hltProcessName_;
  std::string destination_;
  std::vector<std::string> hltPathsToCheck_;

  // Member Variables
  HLTMuonMatchAndPlotContainer plotterContainer_;
  HLTConfigProvider hltConfig_;

  // Access to the DQM
  DQMStore * dbe_;

};



//////////////////////////////////////////////////////////////////////////////
//////// Namespaces, Typedefs, and Constants /////////////////////////////////

using namespace std;
using namespace edm;
using namespace reco;
using namespace trigger;

typedef vector<string> vstring;



//////////////////////////////////////////////////////////////////////////////
//////// Class Methods ///////////////////////////////////////////////////////

HLTMuonOfflineAnalyzer::HLTMuonOfflineAnalyzer(const ParameterSet& pset) :
  pset_(pset),
  hltProcessName_(pset.getParameter<string>("hltProcessName")),
  destination_(pset.getUntrackedParameter<string>("destination")),
  hltPathsToCheck_(pset.getParameter<vstring>("hltPathsToCheck")),
  plotterContainer_(consumesCollector(),pset)
{

  // Prepare the DQMStore object.
  dbe_ = edm::Service<DQMStore>().operator->();
  dbe_->setVerbose(0);
  dbe_->setCurrentFolder(destination_);

}



vector<string> 
HLTMuonOfflineAnalyzer::moduleLabels(string path) 
{

  vector<string> modules = hltConfig_.moduleLabels(path);
  vector<string>::iterator iter = modules.begin();

  while (iter != modules.end())
    if (iter->find("Filtered") == string::npos) 
      iter = modules.erase(iter);
    else
      ++iter;

  return modules;
  
}



void 
HLTMuonOfflineAnalyzer::beginRun(const edm::Run & iRun, 
				 const edm::EventSetup & iSetup) 
{

  // Initialize hltConfig
  bool changedConfig;
  if (!hltConfig_.init(iRun, iSetup, hltProcessName_, changedConfig)) {
    LogError("HLTMuonVal") << "Initialization of HLTConfigProvider failed!!"; 
    return;
  }

  // Get the set of trigger paths we want to make plots for
  set<string> hltPaths;
  for (size_t i = 0; i < hltPathsToCheck_.size(); i++) {
    TPRegexp pattern(hltPathsToCheck_[i]);
    for (size_t j = 0; j < hltConfig_.triggerNames().size(); j++)
      if (TString(hltConfig_.triggerNames()[j]).Contains(pattern))
        hltPaths.insert(hltConfig_.triggerNames()[j]);
  }
  
  // Initialize the plotters
  set<string>::iterator iPath;
  for (iPath = hltPaths.begin(); iPath != hltPaths.end(); iPath++) {
    string path = * iPath;
    vector<string> labels = moduleLabels(path);
    if (labels.size() > 0) {
      plotterContainer_.addPlotter(pset_, path, moduleLabels(path));
    }
  }

  plotterContainer_.beginRun(iRun, iSetup);

}

void
HLTMuonOfflineAnalyzer::analyze(const Event& iEvent, 
				const EventSetup& iSetup)
{

  plotterContainer_.analyze(iEvent, iSetup);

}



void 
HLTMuonOfflineAnalyzer::beginJob()
{
  
}



void 
HLTMuonOfflineAnalyzer::endRun(const edm::Run & iRun, 
			       const edm::EventSetup& iSetup)
{

  //   plotterContainer_.endRun(iRun, iSetup);

}



void 
HLTMuonOfflineAnalyzer::endJob()
{
  
}



//define this as a plug-in
DEFINE_FWK_MODULE(HLTMuonOfflineAnalyzer);
