#ifndef TrackingMonitorClient_TrackingOfflineDQM_h
#define TrackingMonitorClient_TrackingOfflineDQM_h
// -*- C++ -*-
//
// Package:     TrackingMonitorClient
// Class  :     TrackingOfflineDQM
// 
/**\class TrackingOfflineDQM TrackingOfflineDQM.h DQM/TrackingMonitorCluster/interface/TrackingOfflineDQM.h

 Description: 
   DQM class to perform Summary creation Quality Test on a merged Root file
   after CAF processing
 Usage:
    <usage>

*/
//
// Original Author:  Samvel Khalatyan (ksamdev at gmail dot com)
//         Created:  Wed Oct 5 16:47:14 CET 2006
//

#include <string>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class DQMStore;
class TrackingActionExecutor;
class SiStripDetCabling;

class TrackingOfflineDQM: public edm::EDAnalyzer {

 public:

  /// Constructor
  TrackingOfflineDQM(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~TrackingOfflineDQM();

 private:

  /// BeginJob
  void beginJob();

  /// BeginRun
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup);

  /// Analyze
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup);

  /// End Of Luminosity
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& iSetup);

  /// EndRun
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);

  /// Endjob
  void endJob();

private:

  void checkTrackerFEDs(edm::Event const& e);
  bool openInputFile();

  DQMStore* dqmStore_;
  TrackingActionExecutor* actionExecutor_;

  std::string inputFileName_;
  std::string outputFileName_;
  int globalStatusFilling_; 
  bool usedWithEDMtoMEConverter_;
  bool trackerFEDsFound_;
  bool allpixelFEDsFound_;

  edm::ParameterSet configPar_;

};
#endif
