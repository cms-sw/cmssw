#ifndef DQMHistogramStats_H
#define DQMHistogramStats_H

/** \class DQMHistogramStats
 * *
 *  DQM Store Stats - new version, for multithreaded framework
 *
 *  \author Dmitrijus Bugelskis CERN
 */

#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <iomanip>
#include <utility>
#include <fstream>
#include <sstream>

#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMHistoStats.h"

namespace dqmservices {

class DQMHistogramStats : public DQMEDHarvester {
 public:
  DQMHistogramStats(edm::ParameterSet const & iConfig);

  virtual ~DQMHistogramStats();

  // static std::unique_ptr<Stats> initializeGlobalCache(edm::ParameterSet
  // const&);
  // static void globalEndJob(Stats const* iStats);
  // virtual void analyze(edm::Event const&, edm::EventSetup const&) override;
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &,
                             edm::LuminosityBlock const &,
                             edm::EventSetup const &) override {};

  void dqmEndJob(DQMStore::IBooker &iBooker,
                 DQMStore::IGetter &iGetter) override {};

  void dqmEndRun(DQMStore::IBooker &, DQMStore::IGetter &,
              edm::Run const&, 
              edm::EventSetup const&) override {};

  // analyze a single monitor element
  HistoEntry analyze(MonitorElement *m);

  // group summaries per folder
  // void group(HistoStats& st);

 protected:
  HistoStats collect(DQMStore::IGetter &iGetter, const std::set<std::string>& names );
  HistoStats collect(DQMStore::IGetter &iGetter, const std::vector<std::string>& names);
  HistoStats collect(DQMStore::IGetter &iGetter);
  std::string getStepName();
  std::string onlineOfflineFileName(const std::string &fileBaseName, const std::string &suffix, const std::string &workflow, const std::string &child);

  std::string workflow_;
  std::string   child_;
  std::string producer_;
  std::string dirName_;
  std::string fileBaseName_;
  bool dumpOnEndLumi_;
  bool dumpOnEndRun_;
  std::string processName_;

  std::vector<std::string> histogramNamesEndLumi_;
  std::vector<std::string> histogramNamesEndRun_;
  std::set<std::string> histograms_;

 private:
  void getDimensionX(Dimension &d, MonitorElement *m);
  void getDimensionY(Dimension &d, MonitorElement *m);
  void getDimensionZ(Dimension &d, MonitorElement *m);

};

}  // end of namespace
#endif
