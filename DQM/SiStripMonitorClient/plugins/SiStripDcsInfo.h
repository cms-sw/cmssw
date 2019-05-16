#ifndef SiStripMonitorClient_SiStripDcsInfo_h
#define SiStripMonitorClient_SiStripDcsInfo_h
// -*- C++ -*-
//
// Package:     SiStripMonitorClient
// Class  :     SiStripDcsInfo
//
/**\class SiStripDcsInfo SiStripDcsInfo.h
   DQM/SiStripMonitorCluster/interface/SiStripDcsInfo.h

   Description:
   Checks the # of SiStrip FEDs from DAQ
   Usage:
   <usage>

*/
//
//          Author:  Suchandra Dutta
//         Created:  Mon Feb 16 19:00:00 CET 2009
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
#include <unordered_map>

class SiStripDetVOff;
class SiStripDetCabling;

class SiStripDcsInfo : public edm::EDAnalyzer {
public:
  SiStripDcsInfo(const edm::ParameterSet& ps);

private:
  void beginJob() override;
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup) override;
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) override;
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& iSetup) override;
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup) override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;

  void bookStatus(DQMStore& dqm_store);
  void readStatus(edm::EventSetup const&);
  void readCabling(edm::EventSetup const&);
  void addBadModules(DQMStore& dqm_store);
  void fillStatus(DQMStore& dqm_store);
  void fillDummyStatus(DQMStore& dqm_store);

  MonitorElement* DcsFraction_{nullptr};

  struct SubDetMEs {
    std::string folder_name;
    MonitorElement* DcsFractionME;
    int TotalDetectors;
    std::vector<uint32_t> FaultyDetectors;
    std::unordered_map<uint32_t, uint16_t> NLumiDetectorIsFaulty;
  };

  std::map<std::string, SubDetMEs> SubDetMEsMap{};
  unsigned long long m_cacheIDCabling_{};
  unsigned long long m_cacheIDDcs_{};
  bool bookedStatus_{false};

  edm::ESHandle<SiStripDetVOff> siStripDetVOff_{};
  int nFEDConnected_{};

  int nLumiAnalysed_{};

  bool IsLumiGoodDcs_{false};
  int nGoodDcsLumi_{};
  static constexpr float MinAcceptableDcsDetFrac_{0.90};
  static constexpr float MaxAcceptableBadDcsLumi_{2};

  edm::ESHandle<SiStripDetCabling> detCabling_{};
};
#endif
