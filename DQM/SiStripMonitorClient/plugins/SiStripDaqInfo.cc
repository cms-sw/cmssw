#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class SiStripFedCabling;
class TrackerTopology;

class SiStripDaqInfo : public edm::one::EDAnalyzer<edm::one::SharedResources, edm::one::WatchRuns> {
public:
  typedef dqm::harvesting::MonitorElement MonitorElement;
  typedef dqm::harvesting::DQMStore DQMStore;

  SiStripDaqInfo(edm::ParameterSet const& ps);

private:
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup) override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup) override{};

  void readFedIds(const SiStripFedCabling* fedcabling, edm::EventSetup const& iSetup);
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

  int nFedTotal_{};
  bool bookedStatus_{false};

  const SiStripFedCabling* fedCabling_;
  edm::ESWatcher<SiStripFedCablingRcd> fedCablingWatcher_;
  edm::ESGetToken<SiStripFedCabling, SiStripFedCablingRcd> fedCablingToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  edm::ESGetToken<RunInfo, RunInfoRcd> runInfoToken_;
};

SiStripDaqInfo::SiStripDaqInfo(edm::ParameterSet const&)
    : fedCablingToken_{esConsumes<edm::Transition::BeginRun>()},
      tTopoToken_{esConsumes<edm::Transition::BeginRun>()},
      runInfoToken_{esConsumes<edm::Transition::BeginRun>()} {
  usesResource("DQMStore");
  edm::LogInfo("SiStripDaqInfo") << "SiStripDaqInfo::Deleting SiStripDaqInfo ";
}

//
// -- Book MEs for SiStrip Daq Fraction
//
void SiStripDaqInfo::bookStatus(DQMStore& dqm_store) {
  edm::LogInfo("SiStripDcsInfo") << " SiStripDaqInfo::bookStatus " << bookedStatus_;
  if (bookedStatus_)
    return;

  dqm_store.cd();
  std::string strip_dir = "";
  SiStripUtility::getTopFolderPath(dqm_store, "SiStrip", strip_dir);
  if (!strip_dir.empty())
    dqm_store.setCurrentFolder(strip_dir + "/EventInfo");
  else
    dqm_store.setCurrentFolder("SiStrip/EventInfo");

  daqFraction_ = dqm_store.bookFloat("DAQSummary");

  dqm_store.cd();
  if (!strip_dir.empty())
    dqm_store.setCurrentFolder(strip_dir + "/EventInfo/DAQContents");
  else
    dqm_store.setCurrentFolder("SiStrip/EventInfo/DAQContents");

  std::vector<std::string> det_types;
  det_types.push_back("TIB");
  det_types.push_back("TOB");
  det_types.push_back("TIDF");
  det_types.push_back("TIDB");
  det_types.push_back("TECF");
  det_types.push_back("TECB");

  for (auto const& det : det_types) {
    std::string const me_name{"SiStrip_" + det};
    SubDetMEs local_mes{dqm_store.bookFloat(me_name), 0};
    subDetMEsMap_.emplace(det, local_mes);
  }
  bookedStatus_ = true;
  dqm_store.cd();
}
//
// -- Fill with Dummy values
//
void SiStripDaqInfo::fillDummyStatus(DQMStore& dqm_store) {
  if (!bookedStatus_) {
    bookStatus(dqm_store);
  }
  assert(bookedStatus_);

  for (auto& pr : subDetMEsMap_) {
    pr.second.daqFractionME->Reset();
    pr.second.daqFractionME->Fill(-1.0);
  }
  daqFraction_->Reset();
  daqFraction_->Fill(-1.0);
}

void SiStripDaqInfo::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) {
  edm::LogInfo("SiStripDaqInfo") << "SiStripDaqInfo:: Begining of Run";

  // Check latest Fed cabling and create TrackerMapCreator
  if (fedCablingWatcher_.check(eSetup)) {
    fedCabling_ = &eSetup.getData(fedCablingToken_);
    readFedIds(fedCabling_, eSetup);
  }
  auto& dqm_store = *edm::Service<DQMStore>{};
  if (!bookedStatus_) {
    bookStatus(dqm_store);
  }
  if (nFedTotal_ == 0) {
    fillDummyStatus(dqm_store);
    edm::LogInfo("SiStripDaqInfo") << " SiStripDaqInfo::No FEDs Connected!!!";
    return;
  }

  float nFEDConnected = 0.0;
  constexpr int siStripFedIdMin{FEDNumbering::MINSiStripFEDID};
  constexpr int siStripFedIdMax{FEDNumbering::MAXSiStripFEDID};

  if (!eSetup.tryToGet<RunInfoRcd>())
    return;
  auto sumFED = eSetup.getHandle(runInfoToken_);
  if (!sumFED)
    return;

  auto const& fedsInIds = sumFED->m_fed_in;
  for (auto const fedID : fedsInIds) {
    if (fedID >= siStripFedIdMin && fedID <= siStripFedIdMax)
      ++nFEDConnected;
  }
  edm::LogInfo("SiStripDaqInfo") << " SiStripDaqInfo::Total # of FEDs " << nFedTotal_ << " Connected FEDs "
                                 << nFEDConnected;
  if (nFEDConnected > 0) {
    daqFraction_->Reset();
    daqFraction_->Fill(nFEDConnected / nFedTotal_);
    readSubdetFedFractions(dqm_store, fedsInIds, eSetup);
  }
}

void SiStripDaqInfo::analyze(edm::Event const& event, edm::EventSetup const& eSetup) {}

//
// -- Read Sub Detector FEDs
//
void SiStripDaqInfo::readFedIds(const SiStripFedCabling* fedcabling, edm::EventSetup const& iSetup) {
  //Retrieve tracker topology from geometry
  const auto tTopo = &iSetup.getData(tTopoToken_);

  auto feds = fedCabling_->fedIds();

  nFedTotal_ = feds.size();
  for (auto const fed : feds) {
    auto fedChannels = fedCabling_->fedConnections(fed);
    for (auto const& conn : fedChannels) {
      if (!conn.isConnected())
        continue;
      uint32_t detId = conn.detId();
      if (detId == 0 || detId == 0xFFFFFFFF)
        continue;
      std::string subdet_tag;
      SiStripUtility::getSubDetectorTag(detId, subdet_tag, tTopo);
      subDetFedMap_[subdet_tag].push_back(fed);
      break;
    }
  }
}
//
// -- Fill Subdet FEDIds
//
void SiStripDaqInfo::readSubdetFedFractions(DQMStore& dqm_store,
                                            std::vector<int> const& fed_ids,
                                            edm::EventSetup const& iSetup) {
  //Retrieve tracker topology from geometry
  const auto tTopo = &iSetup.getData(tTopoToken_);

  constexpr int siStripFedIdMin{FEDNumbering::MINSiStripFEDID};
  constexpr int siStripFedIdMax{FEDNumbering::MAXSiStripFEDID};

  // initialiase
  for (auto const& pr : subDetFedMap_) {
    auto const& name = pr.first;
    auto iPos = subDetMEsMap_.find(name);
    if (iPos == subDetMEsMap_.end())
      continue;
    iPos->second.connectedFeds = 0;
  }
  // count sub detector feds

  for (auto const& [name, subdetIds] : subDetFedMap_) {
    auto iPos = subDetMEsMap_.find(name);
    if (iPos == subDetMEsMap_.end())
      continue;
    iPos->second.connectedFeds = 0;
    for (auto const subdetId : subdetIds) {
      bool fedid_found = false;
      for (auto const fedId : fed_ids) {
        if (fedId < siStripFedIdMin || fedId > siStripFedIdMax)
          continue;
        if (subdetId == fedId) {
          fedid_found = true;
          iPos->second.connectedFeds++;
          break;
        }
      }
      if (!fedid_found)
        findExcludedModule(dqm_store, subdetId, tTopo);
    }
    if (auto nFedSubDet = subdetIds.size(); nFedSubDet > 0) {
      iPos->second.daqFractionME->Reset();
      int const nFedsConnected = iPos->second.connectedFeds;
      iPos->second.daqFractionME->Fill(nFedsConnected * 1.0 / nFedSubDet);
    }
  }
}
//
// -- find Excluded Modules
//
void SiStripDaqInfo::findExcludedModule(DQMStore& dqm_store,
                                        unsigned short const fed_id,
                                        TrackerTopology const* tTopo) {
  dqm_store.cd();
  std::string mdir = "MechanicalView";
  if (!SiStripUtility::goToDir(dqm_store, mdir)) {
    dqm_store.setCurrentFolder("SiStrip/" + mdir);
  }
  std::string mechanical_dir = dqm_store.pwd();
  auto fedChannels = fedCabling_->fedConnections(fed_id);
  int ichannel = 0;
  std::string tag = "ExcludedFedChannel";
  std::string bad_module_folder;
  for (auto const& conn : fedChannels) {
    if (!conn.isConnected())
      continue;
    uint32_t detId = conn.detId();
    if (detId == 0 || detId == 0xFFFFFFFF)
      continue;

    ichannel++;
    if (ichannel == 1) {
      std::string subdet_folder;
      SiStripFolderOrganizer folder_organizer;
      folder_organizer.getSubDetFolder(detId, tTopo, subdet_folder);
      if (!dqm_store.dirExists(subdet_folder)) {
        subdet_folder = mechanical_dir + subdet_folder.substr(subdet_folder.find(mdir) + mdir.size());
      }
      bad_module_folder = subdet_folder + "/" + "BadModuleList";
      dqm_store.setCurrentFolder(bad_module_folder);
    }
    std::ostringstream detid_str;
    detid_str << detId;
    std::string full_path = bad_module_folder + "/" + detid_str.str();
    MonitorElement* me = dqm_store.get(full_path);
    uint16_t flag = 0;
    if (me) {
      flag = me->getIntValue();
      me->Reset();
    } else
      me = dqm_store.bookInt(detid_str.str());
    SiStripUtility::setBadModuleFlag(tag, flag);
    me->Fill(flag);
  }
  dqm_store.cd();
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripDaqInfo);
