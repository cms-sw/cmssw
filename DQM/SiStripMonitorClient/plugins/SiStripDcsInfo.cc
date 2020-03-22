#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "SiStripDcsInfo.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"

#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

//Run Info
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include <iostream>
#include <iomanip>
#include <cstdio>
#include <string>
#include <sstream>
#include <cmath>

//
// -- Contructor
//
SiStripDcsInfo::SiStripDcsInfo(edm::ParameterSet const& pSet) {
  LogDebug("SiStripDcsInfo") << "SiStripDcsInfo::Deleting SiStripDcsInfo ";
}

void SiStripDcsInfo::beginJob() {
  // Since SubDetMEs is a struct, using the brace initialization will
  // zero-initialize all members that are not specified in the call.
  SubDetMEsMap.emplace("TIB", SubDetMEs{"TIB", nullptr, 0, {}, {}});
  SubDetMEsMap.emplace("TOB", SubDetMEs{"TOB", nullptr, 0, {}, {}});
  SubDetMEsMap.emplace("TECB", SubDetMEs{"TEC/MINUS", nullptr, 0, {}, {}});
  SubDetMEsMap.emplace("TECF", SubDetMEs{"TEC/PLUS", nullptr, 0, {}, {}});
  SubDetMEsMap.emplace("TIDB", SubDetMEs{"TID/MINUS", nullptr, 0, {}, {}});
  SubDetMEsMap.emplace("TIDF", SubDetMEs{"TID/PLUS", nullptr, 0, {}, {}});
}
//
// -- Begin Run
//
void SiStripDcsInfo::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) {
  LogDebug("SiStripDcsInfo") << "SiStripDcsInfo:: Begining of Run";
  nFEDConnected_ = 0;
  constexpr int siStripFedIdMin{FEDNumbering::MINSiStripFEDID};
  constexpr int siStripFedIdMax{FEDNumbering::MAXSiStripFEDID};

  // Count Tracker FEDs from RunInfo

  if (auto runInfoRec = eSetup.tryToGet<RunInfoRcd>()) {
    edm::ESHandle<RunInfo> sumFED;
    runInfoRec->get(sumFED);

    if (sumFED.isValid()) {
      std::vector<int> FedsInIds = sumFED->m_fed_in;
      for (unsigned int it = 0; it < FedsInIds.size(); ++it) {
        int fedID = FedsInIds[it];
        if (fedID >= siStripFedIdMin && fedID <= siStripFedIdMax)
          ++nFEDConnected_;
      }
      LogDebug("SiStripDcsInfo") << " SiStripDcsInfo :: Connected FEDs " << nFEDConnected_;
    }
  }

  auto& dqm_store = *edm::Service<DQMStore>{};
  bookStatus(dqm_store);
  fillDummyStatus(dqm_store);
  if (nFEDConnected_ > 0)
    readCabling(eSetup);
}

void SiStripDcsInfo::analyze(edm::Event const& event, edm::EventSetup const& eSetup) {}

void SiStripDcsInfo::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) {
  LogDebug("SiStripDcsInfo") << "SiStripDcsInfo::beginLuminosityBlock";

  if (nFEDConnected_ == 0)
    return;

  // initialise BadModule list
  for (auto& subDetME : SubDetMEsMap) {
    subDetME.second.FaultyDetectors.clear();
  }
  readStatus(eSetup);
  nLumiAnalysed_++;
}

void SiStripDcsInfo::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) {
  LogDebug("SiStripDcsInfo") << "SiStripDcsInfo::endLuminosityBlock";

  if (nFEDConnected_ == 0)
    return;
  auto& dqm_store = *edm::Service<DQMStore>{};
  readStatus(eSetup);
  fillStatus(dqm_store);
}

void SiStripDcsInfo::endRun(edm::Run const& run, edm::EventSetup const& eSetup) {
  LogDebug("SiStripDcsInfo") << "SiStripDcsInfo::EndRun";

  if (nFEDConnected_ == 0)
    return;

  for (auto& subDetME : SubDetMEsMap) {
    subDetME.second.FaultyDetectors.clear();
  }
  readStatus(eSetup);
  auto& dqm_store = *edm::Service<DQMStore>{};
  addBadModules(dqm_store);
}
//
// -- Book MEs for SiStrip Dcs Fraction
//
void SiStripDcsInfo::bookStatus(DQMStore& dqm_store) {
  if (bookedStatus_)
    return;

  std::string strip_dir = "";
  SiStripUtility::getTopFolderPath(dqm_store, "SiStrip", strip_dir);
  if (!strip_dir.empty())
    dqm_store.setCurrentFolder(strip_dir + "/EventInfo");
  else
    dqm_store.setCurrentFolder("SiStrip/EventInfo");

  auto scope = DQMStore::UseLumiScope(dqm_store);

  DcsFraction_ = dqm_store.bookFloat("DCSSummary");

  dqm_store.cd();
  if (!strip_dir.empty())
    dqm_store.setCurrentFolder(strip_dir + "/EventInfo/DCSContents");
  else
    dqm_store.setCurrentFolder("SiStrip/EventInfo/DCSContents");
  for (auto& [suffix, subDetME] : SubDetMEsMap) {
    std::string const me_name{"SiStrip_" + suffix};
    subDetME.DcsFractionME = dqm_store.bookFloat(me_name);
  }
  bookedStatus_ = true;
  dqm_store.cd();
}

void SiStripDcsInfo::readCabling(edm::EventSetup const& eSetup) {
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  eSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  unsigned long long cacheID = eSetup.get<SiStripFedCablingRcd>().cacheIdentifier();
  if (m_cacheIDCabling_ != cacheID) {
    m_cacheIDCabling_ = cacheID;
    LogDebug("SiStripDcsInfo") << "SiStripDcsInfo::readCabling : "
                               << " Change in Cache";
    eSetup.get<SiStripDetCablingRcd>().get(detCabling_);

    std::vector<uint32_t> SelectedDetIds;
    detCabling_->addActiveDetectorsRawIds(SelectedDetIds);
    LogDebug("SiStripDcsInfo") << " SiStripDcsInfo::readCabling : "
                               << " Total Detectors " << SelectedDetIds.size();

    // initialise total # of detectors first
    for (std::map<std::string, SubDetMEs>::iterator it = SubDetMEsMap.begin(); it != SubDetMEsMap.end(); it++) {
      it->second.TotalDetectors = 0;
    }

    for (std::vector<uint32_t>::const_iterator idetid = SelectedDetIds.begin(); idetid != SelectedDetIds.end();
         ++idetid) {
      uint32_t detId = *idetid;
      if (detId == 0 || detId == 0xFFFFFFFF)
        continue;
      std::string subdet_tag;
      SiStripUtility::getSubDetectorTag(detId, subdet_tag, tTopo);

      std::map<std::string, SubDetMEs>::iterator iPos = SubDetMEsMap.find(subdet_tag);
      if (iPos != SubDetMEsMap.end()) {
        iPos->second.TotalDetectors++;
      }
    }
  }
}
//
// -- Get Faulty Detectors
//
void SiStripDcsInfo::readStatus(edm::EventSetup const& eSetup) {
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  eSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  eSetup.get<SiStripDetVOffRcd>().get(siStripDetVOff_);
  std::vector<uint32_t> FaultyDetIds;
  siStripDetVOff_->getDetIds(FaultyDetIds);
  LogDebug("SiStripDcsInfo") << " SiStripDcsInfo::readStatus : "
                             << " Faulty Detectors " << FaultyDetIds.size();
  // Read and fille bad modules
  for (std::vector<uint32_t>::const_iterator ihvoff = FaultyDetIds.begin(); ihvoff != FaultyDetIds.end(); ++ihvoff) {
    uint32_t detId_hvoff = (*ihvoff);
    if (!detCabling_->IsConnected(detId_hvoff))
      continue;
    std::string subdet_tag;
    SiStripUtility::getSubDetectorTag(detId_hvoff, subdet_tag, tTopo);

    std::map<std::string, SubDetMEs>::iterator iPos = SubDetMEsMap.find(subdet_tag);
    if (iPos != SubDetMEsMap.end()) {
      std::vector<uint32_t>::iterator ibad =
          std::find(iPos->second.FaultyDetectors.begin(), iPos->second.FaultyDetectors.end(), detId_hvoff);
      if (ibad == iPos->second.FaultyDetectors.end())
        iPos->second.FaultyDetectors.push_back(detId_hvoff);
    }
  }
}
//
// -- Fill Status
//
void SiStripDcsInfo::fillStatus(DQMStore& dqm_store) {
  if (!bookedStatus_)
    bookStatus(dqm_store);
  assert(bookedStatus_);

  float total_det = 0.0;
  float faulty_det = 0.0;
  float fraction;
  for (auto const& [name, subDetMEs] : SubDetMEsMap) {
    int total_subdet = subDetMEs.TotalDetectors;
    int faulty_subdet = subDetMEs.FaultyDetectors.size();
    if (nFEDConnected_ == 0 || total_subdet == 0)
      fraction = -1;
    else
      fraction = 1.0 - faulty_subdet * 1.0 / total_subdet;
    subDetMEs.DcsFractionME->Reset();
    subDetMEs.DcsFractionME->Fill(fraction);
    edm::LogInfo("SiStripDcsInfo") << " SiStripDcsInfo::fillStatus : Sub Detector " << name << " Total Number "
                                   << total_subdet << " Faulty ones " << faulty_subdet;
    total_det += total_subdet;
    faulty_det += faulty_subdet;
  }
  if (nFEDConnected_ == 0 || total_det == 0)
    fraction = -1.0;
  else
    fraction = 1 - faulty_det / total_det;
  DcsFraction_->Reset();
  DcsFraction_->Fill(fraction);
  IsLumiGoodDcs_ = fraction > MinAcceptableDcsDetFrac_;
  if (!IsLumiGoodDcs_)
    return;

  ++nGoodDcsLumi_;
  for (auto& pr : SubDetMEsMap) {
    for (auto const detId_faulty : pr.second.FaultyDetectors) {
      pr.second.NLumiDetectorIsFaulty[detId_faulty]++;
    }
  }
}

//
// -- Fill with Dummy values
//
void SiStripDcsInfo::fillDummyStatus(DQMStore& dqm_store) {
  if (!bookedStatus_)
    bookStatus(dqm_store);
  assert(bookedStatus_);

  for (auto& pr : SubDetMEsMap) {
    pr.second.DcsFractionME->Reset();
    pr.second.DcsFractionME->Fill(-1.0);
  }
  DcsFraction_->Reset();
  DcsFraction_->Fill(-1.0);
}

void SiStripDcsInfo::addBadModules(DQMStore& dqm_store) {
  dqm_store.cd();
  std::string mdir = "MechanicalView";
  if (!SiStripUtility::goToDir(dqm_store, mdir)) {
    dqm_store.setCurrentFolder("SiStrip/" + mdir);
  }
  std::string mechanical_dir = dqm_store.pwd();
  std::string tag = "DCSError";

  for (auto const& pr : SubDetMEsMap) {
    auto const& lumiCountBadModules = pr.second.NLumiDetectorIsFaulty;
    for (auto const [ibad, nBadLumi] : lumiCountBadModules) {
      if (nBadLumi <= MaxAcceptableBadDcsLumi_)
        continue;
      std::string bad_module_folder = mechanical_dir + "/" + pr.second.folder_name +
                                      "/"
                                      "BadModuleList";
      dqm_store.setCurrentFolder(bad_module_folder);

      std::ostringstream detid_str;
      detid_str << ibad;
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
  }
  dqm_store.cd();
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripDcsInfo);
