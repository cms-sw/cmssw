#ifndef SiStripMonitorClient_SiStripDaqInfo_h
#define SiStripMonitorClient_SiStripDaqInfo_h
// -*- C++ -*-
//
// Package:     SiStripMonitorClient
// Class  :     SiStripDaqInfo
//
/**\class SiStripDaqInfo SiStripDaqInfo.h
   DQM/SiStripMonitorCluster/interface/SiStripDaqInfo.h

   Description:
   Checks the # of SiStrip FEDs from DAQ
   Usage:
   <usage>

*/
//
//          Author:  Suchandra Dutta
//         Created:  Thu Dec 11 17:50:00 CET 2008
//

#include <string>

#include "FWCore/Framework/interface/EDAnalyzer.h"
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

class SiStripFedCabling;
class TrackerTopology;

class SiStripDaqInfo : public edm::EDAnalyzer {
public:
  typedef dqm::harvesting::MonitorElement MonitorElement;
  typedef dqm::harvesting::DQMStore DQMStore;

  SiStripDaqInfo(edm::ParameterSet const& ps);

private:
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup) override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;

  void readFedIds(edm::ESHandle<SiStripFedCabling> const& fedcabling, edm::EventSetup const& iSetup);
  void readSubdetFedFractions(DQMStore& dqm_store, std::vector<int> const& fed_ids, edm::EventSetup const& iSetup);
  void bookStatus(DQMStore& dqm_store);
  void fillDummyStatus(DQMStore& dqm_store);
  void findExcludedModule(DQMStore& dqm_store, unsigned short fed_id, TrackerTopology const* tTopo);

  std::map<std::string, std::vector<unsigned short>> subDetFedMap_;

  MonitorElement* daqFraction_{nullptr};

  struct SubDetMEs {
    MonitorElement* daqFractionME;
    int connectedFeds;
  };

  std::map<std::string, SubDetMEs> subDetMEsMap_;

  unsigned long long m_cacheID_{};
  int nFedTotal_{};
  bool bookedStatus_{false};

  edm::ESHandle<SiStripFedCabling> fedCabling_;
};
#endif
