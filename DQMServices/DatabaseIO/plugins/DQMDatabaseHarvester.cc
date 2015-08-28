#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMDatabaseHarvester.h"

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

//
// -------------------------------------- Constructor
// --------------------------------------------
//
DQMDatabaseHarvester::DQMDatabaseHarvester(const edm::ParameterSet& ps)
    : DQMEDHarvester() {

  edm::LogInfo("DQMDatabaseHarvester")
      << "Constructor  DQMDatabaseHarvester::DQMDatabaseHarvester "
      << std::endl;
  s_histogramsPath = ps.getParameter<std::string>("histogramsPath");
  vs_histogramsPerLumi =
      ps.getParameter<std::vector<std::string> >("histogramsPerLumi");
  vs_histogramsPerRun =
      ps.getParameter<std::vector<std::string> >("histogramsPerRun");
	  
  dbw_.reset(new DQMDatabaseWriter(ps));
}

//
// -- Destructor
//
DQMDatabaseHarvester::~DQMDatabaseHarvester() {
  edm::LogInfo("DQMDatabaseHarvester")
      << "Destructor DQMDatabaseHarvester::~DQMDatabaseHarvester " << std::endl;
}

//
// --- beginJob ---
// 
void DQMDatabaseHarvester::beginJob() {
	/* if it already exists nothing wrong happens */
  dbw_->initDatabase();
}

//
// --- dqmEndJob ---
//
void DQMDatabaseHarvester::dqmEndJob(DQMStore::IBooker& ibooker_,
                                     DQMStore::IGetter& igetter_) {
}

//
// --- dqmEndLuminosityBlock ---
//
void DQMDatabaseHarvester::dqmEndLuminosityBlock(
    DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_,
    edm::LuminosityBlock const& iLumi, edm::EventSetup const& iSetup) {
  edm::LogInfo("DQMDatabaseHarvester") << "DQMDatabaseHarvester::dqmEndLuminosityBlock " << std::endl;

  if (histogramsPerLumi.empty()) {
    for (std::string histogramName : vs_histogramsPerLumi) {
      MonitorElement* histogram =
          igetter_.get(s_histogramsPath + histogramName);
      if (histogram) {
        histogramsPerLumi.push_back(histogram);
      }
      /* discard repetitions */
      auto sortFunction = [](auto a, auto b) -> bool {
        return a->getName().compare(b->getName());
      };
      sort(histogramsPerLumi.begin(), histogramsPerLumi.end(), sortFunction);
      auto uniqueFunction =
          [](auto a, auto b) -> bool { return a->getName() == b->getName(); };
      histogramsPerLumi.erase(unique(histogramsPerLumi.begin(),
                                     histogramsPerLumi.end(), uniqueFunction),
                              histogramsPerLumi.end());
    }
  }

  // Parse histograms that should be treated as run based
  // It is necessary to gather data from every lumi, so it cannot be done in
  // the endRun
  if (histogramsPerRun.empty()) {
    for (std::string histogramName : vs_histogramsPerRun) {
      HistogramValues histogramValues;
      MonitorElement* histogram =
          igetter_.get(s_histogramsPath + histogramName);
      if (histogram)
        histogramsPerRun.push_back(std::pair<MonitorElement*, HistogramValues>(
            histogram, histogramValues));

      /* discard repetitions */
      auto sortFunction = [](auto a, auto b) -> bool {
        return a.first->getName().compare(b.first->getName());
      };
      sort(histogramsPerRun.begin(), histogramsPerRun.end(), sortFunction);
      auto uniqueFunction = [](auto a, auto b) -> bool {
        return a.first->getName() == b.first->getName();
      };
      histogramsPerRun.erase(unique(histogramsPerRun.begin(),
                                    histogramsPerRun.end(), uniqueFunction),
                             histogramsPerRun.end());
    }
    dbw_->dqmDbRunInitialize(histogramsPerRun);
  }
  dbw_->dqmDbLumiDrop(histogramsPerLumi, iLumi.luminosityBlock(), iLumi.run());
}

//
// --- endRun ---
//
void DQMDatabaseHarvester::endRun(edm::Run const& run,
                                  edm::EventSetup const& eSetup) {
  edm::LogInfo("DQMDatabaseHarvester") <<  "DQMDatabaseHarvester::endRun" << std::endl;
  dbw_->dqmDbRunDrop();
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DQMDatabaseHarvester);
