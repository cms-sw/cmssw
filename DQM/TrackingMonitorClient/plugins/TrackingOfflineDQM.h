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

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class TrackingActionExecutor;
class SiStripDetCabling;
class RunInfo;
class RunInfoRcd;

class TrackingOfflineDQM : public DQMEDHarvester {
public:
  /// Constructor
  TrackingOfflineDQM(const edm::ParameterSet& ps);

  /// Destructor
  ~TrackingOfflineDQM() override;

private:
  /// BeginJob
  void beginJob() override;

  /// BeginRun
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup) override;

  /// End Luminosity Block
  void dqmEndLuminosityBlock(DQMStore::IBooker& ibooker_,
                             DQMStore::IGetter& igetter_,
                             edm::LuminosityBlock const& lumiSeg,
                             edm::EventSetup const& eSetup) override;

  /// Endjob
  void dqmEndJob(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_) override;

private:
  bool openInputFile();

  TrackingActionExecutor* actionExecutor_;
  std::string inputFileName_;
  std::string outputFileName_;
  int globalStatusFilling_;
  bool usedWithEDMtoMEConverter_;
  bool trackerFEDsFound_;
  bool allpixelFEDsFound_;

  edm::ParameterSet configPar_;
  edm::ESGetToken<RunInfo, RunInfoRcd> runInfoToken_;
  const RunInfo* sumFED_ = nullptr;
};
#endif
