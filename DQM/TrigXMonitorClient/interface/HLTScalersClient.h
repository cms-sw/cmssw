// -*-c++-*-
//
// Client class for HLT Scalers module.
//

// Revision 1.13  2010/03/17 20:56:18  wittich
// Check for good updates based on mergeCount values
// add code for rates normalized per FU
//
// Revision 1.12  2010/03/16 22:19:19  wittich
// updates for per-LS normalization for variable
// number of FU's sending information back to the clients.
//
// Revision 1.11  2010/02/15 17:10:45  wittich
// Allow for longer length runs (2400 ls)
// this is only in the client
//
// Revision 1.10  2010/02/11 23:55:18  wittich
// - adapt to shorter Lumi Section length
// - fix bug in how history of counts was filled
//
// Revision 1.9  2010/02/11 00:11:09  wmtan
// Adapt to moved framework header
//
// Revision 1.8  2010/02/02 11:44:20  wittich
// more diagnostics for online scalers
//
// Revision 1.7  2009/12/15 20:41:16  wittich
// better hlt scalers client
//
// Revision 1.6  2009/11/22 14:17:46  puigh
// fix compilation warning
//
// Revision 1.5  2009/11/22 13:32:38  puigh
// clean beginJob
//

#ifndef HLTSCALERSCLIENT_H
#define HLTSCALERSCLIENT_H
#include <deque>
#include <fstream>
#include <utility>
#include <vector>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Utilities/interface/InputTag.h"
//#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#define MAX_PATHS 200
#define MAX_LUMI_SEG_HLT 2400

class HLTScalersClient
    : public edm::one::EDAnalyzer<edm::one::SharedResources, edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
private:
  std::ofstream textfile_;

public:
  typedef dqm::legacy::MonitorElement MonitorElement;
  typedef dqm::legacy::DQMStore DQMStore;
  //   typedef std::pair<int,double> CountLS_t;
  //   //typedef std::deque<CountLS_t> CountLSFifo_t;
  //   typedef std::map<int,double> CountLSFifo_t;

  // helper data structures - slightly modified stl objects
  class CountLS_t : public std::pair<int, double> {
  public:
    CountLS_t(int ls, double cnt) : std::pair<int, double>(ls, cnt){};
    bool operator==(int ls) const { return ls == this->first; }
    bool operator<(CountLS_t &rhs) { return this->first < rhs.first; };
  };

  class CountLSFifo_t : public std::deque<CountLS_t> {
  private:
    unsigned int targetSize_;
    bool accumulate_;

  public:
    // default constructor
    CountLSFifo_t(unsigned int sz = 3) : std::deque<CountLS_t>(), targetSize_(sz) {}
    unsigned int targetSize() const { return targetSize_; };
    double getCount(int ls) {
      CountLSFifo_t::iterator p = std::find(this->begin(), this->end(), ls);
      if (p != end())
        return p->second;
      else
        return -1;
    }

    void update(const CountLS_t &T) {
      // do we already have data for this LS?
      CountLSFifo_t::iterator p = std::find(this->begin(), this->end(), T.first);
      if (p != this->end()) {  // we already have data for this LS
        p->second = T.second;
      } else {  // new data
        this->push_back(T);
      }
      trim_();
    }

  private:
    void trim_() {
      if (this->size() > targetSize_) {
        std::sort(begin(), end());
        while (size() > targetSize_) {
          pop_front();
        }
      }
    }
  };

public:
  /// Constructors
  HLTScalersClient(const edm::ParameterSet &ps);

  /// Destructor
  ~HLTScalersClient() override {
    if (debug_) {
      textfile_.close();
    }
  };

  /// BeginJob
  void beginJob(void) override;

  /// BeginRun
  void beginRun(const edm::Run &run, const edm::EventSetup &c) override;

  /// EndRun
  void endRun(const edm::Run &run, const edm::EventSetup &c) override;

  /// End LumiBlock
  /// DQM Client Diagnostic should be performed here
  void beginLuminosityBlock(const edm::LuminosityBlock &lumiSeg, const edm::EventSetup &c) override {}
  void endLuminosityBlock(const edm::LuminosityBlock &lumiSeg, const edm::EventSetup &c) override;

  // unused
  void analyze(const edm::Event &e, const edm::EventSetup &c) override;

private:
  DQMStore *dbe_;

  int nev_;    // Number of events processed
  int nLumi_;  // number of lumi blocks
  int currentRun_;

  MonitorElement *currentRate_;
  int currentLumiBlockNumber_;
  std::vector<MonitorElement *> rateHistories_;
  std::vector<MonitorElement *> countHistories_;

  std::vector<MonitorElement *> hltCurrentRate_;
  MonitorElement *hltRate_;   // global rate - any accept
  MonitorElement *hltCount_;  // globalCounts
  //  MonitorElement *hltCountN_; // globalCounts normalized
  MonitorElement *updates_;
  MonitorElement *mergeCount_;

  // Normalized
  MonitorElement *hltNormRate_;  // global rate - any accept
  MonitorElement *currentNormRate_;
  std::vector<MonitorElement *> rateNormHistories_;
  std::vector<MonitorElement *> hltCurrentNormRate_;

  bool first_, missingPathNames_;
  std::string folderName_;
  unsigned int kRateIntegWindow_;
  std::string processName_;
  // HLTConfigProvider hltConfig_;
  std::deque<int> ignores_;
  std::pair<double, double> getSlope_(const CountLSFifo_t &points);

private:
  bool debug_;
  int maxFU_;
  std::vector<CountLSFifo_t> recentPathCountsPerLS_;
  CountLSFifo_t recentOverallCountsPerLS_;

  std::vector<CountLSFifo_t> recentNormedPathCountsPerLS_;
  CountLSFifo_t recentNormedOverallCountsPerLS_;
};

#endif  // HLTSCALERSCLIENT_H
