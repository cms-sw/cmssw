// -*-c++-*-
#ifndef L1Scalers_H
#define L1Scalers_H
// $Id: L1Scalers.h,v 1.15 2010/05/31 14:43:43 wteo Exp $

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Utilities/interface/InputTag.h"

#define MAX_LUMI_SEG 2000
#define MAX_LUMI_BIN 400

class L1Scalers: public edm::EDAnalyzer
{
public:
  /// Constructors
  L1Scalers(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~L1Scalers() {};
  
  /// BeginJob
  void beginJob(void);

   // Endjob
   void endJob(void);
  
  /// BeginRun
  void beginRun(const edm::Run& run, const edm::EventSetup& c);

  /// EndRun
  void endRun(const edm::Run& run, const edm::EventSetup& c);

  

  /// End LumiBlock
  /// DQM Client Diagnostic should be performed here
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c);

  void analyze(const edm::Event& e, const edm::EventSetup& c) ;


private:
  DQMStore * dbe_;
  int nev_; // Number of events processed

  bool verbose_;
  edm::InputTag l1GtDataSource_; // L1 Scalers
  
  bool denomIsTech_;
  unsigned int denomBit_;
  bool tfIsTech_;
  unsigned int tfBit_;
  std::vector<unsigned int> algoSelected_;
  std::vector<unsigned int> techSelected_;

  std::string folderName_; // dqm folder name
  MonitorElement *l1scalers_;
  MonitorElement *l1techScalers_;
  MonitorElement *l1Correlations_;
  MonitorElement *bxNum_;


  // 2d versions
  MonitorElement *l1scalersBx_;
  MonitorElement *l1techScalersBx_;
//   MonitorElement *pixFedSizeBx_;
//   MonitorElement *hfEnergyMaxTowerBx_;

  // Int
  MonitorElement *nLumiBlock_;
  MonitorElement *l1AlgoCounter_;  //for total Algo Rate
  MonitorElement *l1TtCounter_;    //for total TT Rate

  //timing plots
  std::vector<MonitorElement* > algoBxDiff_;
  std::vector<MonitorElement* > techBxDiff_;
  std::vector<MonitorElement* > algoBxDiffLumi_;
  std::vector<MonitorElement* > techBxDiffLumi_;
  MonitorElement *dtBxDiff_;
  MonitorElement *dtBxDiffLumi_;
  MonitorElement *cscBxDiff_;
  MonitorElement *cscBxDiffLumi_;
  MonitorElement *rpcbBxDiff_;
  MonitorElement *rpcbBxDiffLumi_;
  MonitorElement *rpcfBxDiff_;
  MonitorElement *rpcfBxDiffLumi_;

  // Hacks for early running
//   MonitorElement *pixFedSize_;
//   MonitorElement *hfEnergy_;
  // steal from HLTrigger/special
  unsigned int threshold_;
  unsigned int fedStart_, fedStop_ ;
  //total Rates
  unsigned int rateAlgoCounter_;  //for total Algo Rate
  unsigned int rateTtCounter_;     //for total TT Rate
 

  edm::InputTag fedRawCollection_;

  std::vector<int> maskedList_;
  edm::InputTag HcalRecHitCollection_;
  // END HACK

  int earliestDenom_;
  std::vector<int> earliestTech_;
  std::vector<int> earliestAlgo_;

};

#endif // L1Scalers_H
