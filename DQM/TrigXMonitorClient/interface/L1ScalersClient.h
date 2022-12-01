// -*-c++-*-
//
// Client class for L1 Scalers module.
//

#ifndef L1ScalersCLIENT_H
#define L1ScalersCLIENT_H

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Utilities/interface/InputTag.h"

// HARD CODE THE NUMBER OF HISTOGRAMS TO 200, LENGTH OF MONITOR to 2000
// segments
#define MAX_ALGOS 140
#define MAX_TT 80
#define MAX_LUMI_SEG 2000

class L1ScalersClient
    : public edm::one::EDAnalyzer<edm::one::SharedResources, edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
public:
  typedef dqm::legacy::MonitorElement MonitorElement;
  typedef dqm::legacy::DQMStore DQMStore;

  /// Constructors
  L1ScalersClient(const edm::ParameterSet &ps);

  /// Destructor
  ~L1ScalersClient() override{};

  /// BeginJob
  void beginJob(void) override;

  //   /// Endjob
  //   void endJob(void);

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

  float totAlgoPrevCount;
  float totTtPrevCount;

  unsigned long int l1AlgoScalerCounters_[MAX_ALGOS];
  MonitorElement *l1AlgoCurrentRate_;
  MonitorElement *l1AlgoRateHistories_[MAX_ALGOS];  // HARD CODE FOR NOW
  MonitorElement *l1AlgoCurrentRatePerAlgo_[MAX_ALGOS];
  MonitorElement *totalAlgoRate_;

  unsigned long int l1TechTrigScalerCounters_[MAX_TT];
  MonitorElement *l1TechTrigCurrentRate_;
  MonitorElement *l1TechTrigRateHistories_[MAX_TT];  // HARD CODE FOR NOW
  MonitorElement *l1TechTrigCurrentRatePerAlgo_[MAX_TT];
  MonitorElement *totalTtRate_;

  // this is a selected list of guys
  MonitorElement *selected_;
  MonitorElement *bxSelected_;
  std::vector<int> algoSelected_;
  std::vector<int> techSelected_;
  std::string folderName_;
  int numSelected_;

  int currentLumiBlockNumber_;
  bool first_algo;
  bool first_tt;
};

#endif  // L1ScalersCLIENT_H
