// -*- C++ -*-
//
// Package:    LogMessageMonitor
// Class:      LogMessageMonitor
//
/**\class LogMessageMonitor LogMessageMonitor.cc DQM/LogMonitor/src/LogMessageMonitor.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Mia Tosi,40 3-B32,+41227671609,
//         Created:  Thu Mar  8 14:34:13 CET 2012
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/ErrorSummaryEntry.h"

#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include <vector>
#include <string>
#include <map>

class GenericTriggerEventFlag;

class GetLumi;

//
// class declaration
//

class LogMessageMonitor : public DQMEDAnalyzer {
public:
  explicit LogMessageMonitor(const edm::ParameterSet&);
  ~LogMessageMonitor() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  //      virtual void beginJob() ;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<std::vector<edm::ErrorSummaryEntry> > errorToken_;

  std::string histname;  //for naming the histograms according to algorithm used

  DQMStore* dqmStore_;
  edm::ParameterSet conf_;

  std::map<std::string, int> modulesMap;

  // from parameters
  std::string pluginsMonName_;
  std::vector<std::string> modules_vector_;
  std::vector<std::string> categories_vector_;

  GetLumi* lumiDetails_;
  GenericTriggerEventFlag* genTriggerEventFlag_;

  // MEs
  std::vector<MonitorElement*> ModulesErrorsVsBXlumi;
  std::vector<MonitorElement*> ModulesWarningsVsBXlumi;

  MonitorElement* CategoriesVsModules;

  bool doWarningsPlots_;
  bool doPUmonitoring_;
};
