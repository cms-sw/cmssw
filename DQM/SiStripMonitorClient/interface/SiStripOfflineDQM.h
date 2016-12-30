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

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class SiStripActionExecutor;
class SiStripDetCabling;

class SiStripOfflineDQM: public DQMEDHarvester {

 public:

  /// Constructor
  SiStripOfflineDQM(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~SiStripOfflineDQM();

 private:

  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup);
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);

  void dqmEndLuminosityBlock(DQMStore::IBooker & , DQMStore::IGetter & , edm::LuminosityBlock const &, edm::EventSetup const&);
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

private:

  SiStripActionExecutor* actionExecutor_;

  bool usedWithEDMtoMEConverter_;
  bool createSummary_;
  int globalStatusFilling_; 
  bool trackerFEDsFound_;
  bool printFaultyModuleList_;
  edm::ESHandle< SiStripDetCabling > det_cabling;
  const TrackerTopology *tTopo;
  edm::ParameterSet configPar_;
  edm::ESHandle<SiStripQuality> ssq;
  bool useSSQuality_;
  std::string ssqLabel_;
  bool configRead;

};
#endif
