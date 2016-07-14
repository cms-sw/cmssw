#ifndef DQMStreamStats_H
#define DQMStreamStats_H

/** \class DQMStreamStats
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

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>



namespace dqmservices {

struct Node{
  std::string name;
  long size;
  std::string parent;
  std::vector<Node> children;
};

struct Dimension{
  int nBin = 0;
  double low = 0, up = 0;
  double mean = 0, meanError = 0; 
  double rms = 0, rmsError = 0; 
  double underflow = 0, overflow = 0;  
};

class HistoEntry {
 public:
  std::string path;

  std::string name;
  const char *type;
  size_t bin_count = 0;
  size_t bin_size = 0;
  size_t extra = 0;
  size_t total = 0;
  double entries = 0;
  int maxBin = 0, minBin = 0;
  double maxValue = 0, minValue = 0;
  Dimension dimX, dimY, dimZ; 

  bool operator<(const HistoEntry &rhs) const { return path < rhs.path; }
};

typedef std::set<HistoEntry> HistoStats;

class DQMStreamStats : public DQMEDHarvester {
 public:
  DQMStreamStats(edm::ParameterSet const & iConfig);

  virtual ~DQMStreamStats();

  // static std::unique_ptr<Stats> initializeGlobalCache(edm::ParameterSet
  // const&);
  // static void globalEndJob(Stats const* iStats);
  // virtual void analyze(edm::Event const&, edm::EventSetup const&) override;
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &,
                             edm::LuminosityBlock const &,
                             edm::EventSetup const &) override;

  void dqmEndJob(DQMStore::IBooker &iBooker,
                 DQMStore::IGetter &iGetter) override{};

  void dqmEndRun(DQMStore::IBooker &, DQMStore::IGetter &,
              edm::Run const&, 
              edm::EventSetup const&) override;

  // analyze a single monitor element
  HistoEntry analyze(MonitorElement *m);

  // group summaries per folder
  // void group(HistoStats& st);

 protected:
  HistoStats collect(DQMStore::IGetter &iGetter);
  void writeMemoryJson(const std::string &fn, const HistoStats &stats);

 private:
  std::string onlineOfflineFileName(const std::string &fileBaseName, const std::string &suffix,
                                        const std::string &workflow, const std::string &child);
  std::string toString(boost::property_tree::ptree doc);
  std::string getStepName();
  void getDimensionX(Dimension &d, MonitorElement *m);
  void getDimensionY(Dimension &d, MonitorElement *m);
  void getDimensionZ(Dimension &d, MonitorElement *m);

  std::string workflow_;
  std::string   child_;
  std::string producer_;
  std::string dirName_;
  std::string fileBaseName_;
  bool dumpOnEndLumi_;
  bool dumpOnEndRun_;
  std::string processName_;
};

}  // end of namespace
#endif
