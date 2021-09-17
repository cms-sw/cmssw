#ifndef SiStripMonitorClient_SiStripCertificationInfo_h
#define SiStripMonitorClient_SiStripCertificationInfo_h
// -*- C++ -*-
//
// Package:     SiStripMonitorClient
// Class  :     SiStripCertificationInfo
//
/**\class SiStripCertificationInfo SiStripCertificationInfo.h
   DQM/SiStripMonitorCluster/interface/SiStripCertificationInfo.h

   Description:
   Checks the # of SiStrip FEDs from DAQ
   Usage:
   <usage>

*/
//
//          Author:  Suchandra Dutta
//         reated:  Mon Feb 16 19200:00 CET 2009
//

#include <string>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class SiStripDetCabling;
class RunInfo;

class SiStripCertificationInfo : public edm::EDAnalyzer {
public:
  typedef dqm::harvesting::MonitorElement MonitorElement;
  typedef dqm::harvesting::DQMStore DQMStore;

  SiStripCertificationInfo(const edm::ParameterSet& ps);

private:
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup) override;
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& iSetup) override;
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup) override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;

  void bookSiStripCertificationMEs(DQMStore& dqm_store);
  void resetSiStripCertificationMEs(DQMStore& dqm_store);
  void fillSiStripCertificationMEs(DQMStore& dqm_store, edm::EventSetup const& eSetup);

  void fillDummySiStripCertification(DQMStore& dqm_store);
  void fillSiStripCertificationMEsAtLumi(DQMStore& dqm_store);

  struct SubDetMEs {
    MonitorElement* det_fractionME;
    std::string folder_name;
    std::string subdet_tag;
    int n_layer;
  };

  MonitorElement* SiStripCertification{nullptr};
  MonitorElement* SiStripCertificationMap{nullptr};
  std::map<std::string, SubDetMEs> SubDetMEsMap{};
  MonitorElement* SiStripCertificationSummaryMap{nullptr};

  bool sistripCertificationBooked_{false};

  edm::ESHandle<SiStripDetCabling> detCabling_{};

  int nFEDConnected_{};

  edm::ESGetToken<SiStripDetCabling, SiStripDetCablingRcd> detCablingToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  edm::ESGetToken<RunInfo, RunInfoRcd> runInfoToken_;
};
#endif
