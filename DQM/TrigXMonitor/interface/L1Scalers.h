// -*-c++-*-
#ifndef L1Scalers_H
#define L1Scalers_H
// $Id$

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

class L1Scalers: public edm::EDAnalyzer
{
public:
  /// Constructors
  L1Scalers(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~L1Scalers() {};
  
  /// BeginJob
  void beginJob(const edm::EventSetup& c);

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
  edm::InputTag scalersSource_;
  
  std::string outputFile_;	//file name for ROOT ouput
  bool verbose_, monitorDaemon_;
  edm::InputTag l1GtDataSource_; // L1 Scalers
  MonitorElement *l1scalers_;
  MonitorElement *l1techScalers_;
  MonitorElement *l1Correlations_;
  MonitorElement *bxNum_;

  // 2d versions
  MonitorElement *l1scalersBx_;
  MonitorElement *l1techScalersBx_;


};

#endif // L1Scalers_H
