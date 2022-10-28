// -*- C++ -*-
//
// Package:    StatisticsFilter
// Class:      StatisticsFilter
//
/**\class StatisticsFilter StatisticsFilter.cc MyFilter/StatisticsFilter/src/StatisticsFilter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Gordon Kaussen,40 1-A15,+41227671647,
//         Created:  Mon Nov 15 10:48:54 CET 2010
// $Id$
//
//

// system include files
#include <memory>

// user include files
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//
// class declaration
//

class StatisticsFilter : public edm::stream::EDFilter<> {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  explicit StatisticsFilter(const edm::ParameterSet&);
  ~StatisticsFilter() override = default;

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------

  DQMStore* dqmStore_;

  std::string filename, dirpath;
  int TotNumberOfEvents;
  int MinNumberOfEvents;
};

//
// constructors and destructor
//
StatisticsFilter::StatisticsFilter(const edm::ParameterSet& iConfig)
    : filename(iConfig.getUntrackedParameter<std::string>("rootFilename", "")),
      dirpath(iConfig.getUntrackedParameter<std::string>("histoDirPath", "")),
      MinNumberOfEvents(iConfig.getUntrackedParameter<int>("minNumberOfEvents")) {
  //now do what ever initialization is needed

  dqmStore_ = edm::Service<DQMStore>().operator->();
  dqmStore_->open(filename, false);
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool StatisticsFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  TotNumberOfEvents = 0;

  std::vector<MonitorElement*> MEs = dqmStore_->getAllContents(dirpath);

  std::vector<MonitorElement*>::const_iterator iter = MEs.begin();
  std::vector<MonitorElement*>::const_iterator iterEnd = MEs.end();

  for (; iter != iterEnd; ++iter) {
    std::string me_name = (*iter)->getName();

    if (strstr(me_name.c_str(), "TotalNumberOfCluster__T") != nullptr &&
        strstr(me_name.c_str(), "Profile") == nullptr) {
      TotNumberOfEvents = ((TH1F*)(*iter)->getTH1F())->GetEntries();

      break;
    }
  }

  if (TotNumberOfEvents < MinNumberOfEvents) {
    edm::LogInfo("StatisticsFilter") << "Only " << TotNumberOfEvents << " events in the run. Run will not be analyzed!";

    return false;
  }

  return true;
}

//define this as a plug-in
DEFINE_FWK_MODULE(StatisticsFilter);
