// -*- C++ -*-
//
// Package:     HLTMuonValidator
// Class:       HLTMuonValidator
// 

//
// Jason Slaunwhite and Jeff Klukas
//
//

// system include files
#include <memory>
#include <iostream>

// user include files
#include "HLTriggerOffline/Muon/interface/HLTMuonPlotter.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


#include "TFile.h"
#include "TDirectory.h"
#include "TPRegexp.h"
#include "boost/tuple/tuple.hpp"



//////////////////////////////////////////////////////////////////////////////
//////// Define the interface ////////////////////////////////////////////////



class HLTMuonValidator : public DQMEDAnalyzer {

public:

  explicit HLTMuonValidator(const edm::ParameterSet&);

private:

  // Analyzer Methods
  virtual void beginJob();
  virtual void dqmBeginRun(const edm::Run &, const edm::EventSetup &) override;
  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  virtual void analyze(const edm::Event &, const edm::EventSetup &) override;
  virtual void endRun(const edm::Run &, const edm::EventSetup &) override;
  virtual void endJob();

  // Extra Methods
  std::vector<std::string> moduleLabels(std::string);
  std::vector<std::string> stepLabels(const std::vector<std::string>&);

  // Input from Configuration File
  edm::ParameterSet pset_;
  std::string hltProcessName_;
  std::vector<std::string> hltPathsToCheck_;

  // Member Variables
  std::vector<HLTMuonPlotter> analyzers_;
  HLTConfigProvider hltConfig_;
  boost::tuple<
    edm::EDGetTokenT<trigger::TriggerEventWithRefs>,
    edm::EDGetTokenT<reco::GenParticleCollection>,
    edm::EDGetTokenT<reco::MuonCollection> > myTokens_;

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

HLTMuonValidator::HLTMuonValidator(const ParameterSet& pset) :
  pset_(pset),
  hltProcessName_(pset.getParameter<string>("hltProcessName")),
  hltPathsToCheck_(pset.getParameter<vstring>("hltPathsToCheck"))
{

  myTokens_ = HLTMuonPlotter::getTokens(pset_, consumesCollector());

}



vector<string> 
HLTMuonValidator::moduleLabels(string path) {

  vector<string> modules = hltConfig_.moduleLabels(path);
  vector<string>::iterator iter = modules.begin();

  while (iter != modules.end())
    if (iter->find("Filtered") == string::npos) 
      iter = modules.erase(iter);
    else
      ++iter;

  return modules;

}



vector<string>
HLTMuonValidator::stepLabels(const vector<string>& modules) {
    vector<string> steps(1, "All");
    for (size_t i = 0; i < modules.size(); i++) {
        if ((modules[i].find("IsoFiltered") != string::npos)){
            if (modules[i].find("L3") != string::npos)
                steps.push_back("L3TkIso");
            else
                steps.push_back("L2Iso");
        }
        else if ((modules[i].find("pfecalIsoRhoFiltered") != string::npos)||(modules[i].find("pfecalOldIsoRhoFiltered") != string::npos)) {
            if (modules[i].find("L3") != string::npos)
                steps.push_back("L3EcalIso");
            else if (modules[i].find("TkFiltered") != string::npos)
                steps.push_back("TkEcalIso");
        }
        else if ((modules[i].find("pfhcalIsoRhoFiltered") != string::npos)||(modules[i].find("pfhcalOldIsoRhoFiltered") != string::npos)) {
            if (modules[i].find("L3") != string::npos)
                steps.push_back("L3HcalIso");
            else if (modules[i].find("TkFiltered") != string::npos)
                steps.push_back("TkHcalIso");
        }
        else if (modules[i].find("TkFiltered") != string::npos){
            steps.push_back("Tk");
        }
        else if (modules[i].find("L3") != string::npos)
            steps.push_back("L3");
        else if (modules[i].find("L2") != string::npos)
            steps.push_back("L2");
        else if (modules[i].find("L1") != string::npos)
            steps.push_back("L1");
        else
            return vector<string>();
    }

    if (steps.size() < 2 || ((steps[1] != "L1")&&(steps[1] != "Tk")))
        return vector<string>();
    return steps;
    
}



void 
HLTMuonValidator::dqmBeginRun(const edm::Run & iRun, 
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
  
  // Initialize the analyzers
  analyzers_.clear();
  set<string>::iterator iPath;
  for (iPath = hltPaths.begin(); iPath != hltPaths.end(); iPath++) {
    
    string path = * iPath;
    string shortpath = path;
    if (path.rfind("_v") < path.length())
      shortpath = path.substr(0, path.rfind("_v"));
    
    vector<string> labels = moduleLabels(path);
    vector<string> steps = stepLabels(labels);

    if (labels.size() > 0 && steps.size() > 0) {
      HLTMuonPlotter analyzer(pset_, shortpath, labels, steps, myTokens_);
      analyzers_.push_back(analyzer);
    }
  }

}



void HLTMuonValidator::bookHistograms(DQMStore::IBooker & iBooker, 
				      edm::Run const & iRun, 
				      edm::EventSetup const & iSetup)
{

  // Call the beginRun (which books all the histograms)
  vector<HLTMuonPlotter>::iterator iter;
  for (iter = analyzers_.begin(); iter != analyzers_.end(); ++iter) {
    iter->beginRun(iBooker, iRun, iSetup);
  }
  
}



void
HLTMuonValidator::analyze(const Event& iEvent, 
			  const EventSetup& iSetup)
{
  
  vector<HLTMuonPlotter>::iterator iter;
  for (iter = analyzers_.begin(); iter != analyzers_.end(); ++iter) {
    iter->analyze(iEvent, iSetup);
  }
  
}



void 
HLTMuonValidator::beginJob()
{

}



void 
HLTMuonValidator::endRun(const edm::Run & iRun, 
			 const edm::EventSetup& iSetup)
{

  // vector<HLTMuonPlotter>::iterator iter;
  // for (iter = analyzers_.begin(); iter != analyzers_.end(); ++iter) {
  //   iter->endRun(iRun, iSetup);
  // }

}



void 
HLTMuonValidator::endJob()
{

}



//define this as a plug-in
DEFINE_FWK_MODULE(HLTMuonValidator);
