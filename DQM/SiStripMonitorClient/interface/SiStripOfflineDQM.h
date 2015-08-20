#ifndef SiStripMonitorClient_SiStripOfflineDQM_h
#define SiStripMonitorClient_SiStripOfflineDQM_h
// -*- C++ -*-
//
// Package:     SiStripMonitorClient
// Class  :     SiStripOfflineDQM
// 
/**\class SiStripOfflineDQM SiStripOfflineDQM.h DQM/SiStripMonitorCluster/interface/SiStripOfflineDQM.h

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
#include <TTree.h>

class DQMStore;
class SiStripActionExecutor;
class SiStripDetCabling;

class SiStripOfflineDQM: public edm::EDAnalyzer {

 public:

  /// Constructor
  SiStripOfflineDQM(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~SiStripOfflineDQM();

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

  SiStripActionExecutor* actionExecutor_;

  bool createSummary_;
  bool createTkInfoFile_;
  std::string inputFileName_;
  int globalStatusFilling_; 
  bool usedWithEDMtoMEConverter_;
  int nEvents_;
  bool trackerFEDsFound_;
  bool printFaultyModuleList_;
  TTree* tkinfoTree_;

  edm::ParameterSet configPar_;

};
#endif
