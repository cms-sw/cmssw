#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/TkHistoMap.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQM/SiStripMonitorClient/interface/SiStripConfigParser.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerNumberingBuilder/interface/utils.h"
#include "DQM/SiStripMonitorClient/interface/SiStripTrackerMapCreator.h"

#include <iostream>

SiStripTrackerMapCreator::SiStripTrackerMapCreator(const edm::EventSetup& eSetup) : eSetup_{eSetup} {
  eSetup_.get<SiStripDetCablingRcd>().get(detCabling_);
  edm::ESHandle<TkDetMap> tkDetMapHandle;
  eSetup_.get<TrackerTopologyRcd>().get(tkDetMapHandle);
  tkDetMap_ = tkDetMapHandle.product();
}

//
// -- Create Geometric and Fed Tracker Map
//
void SiStripTrackerMapCreator::create(const edm::ParameterSet& tkmapPset,
                                      DQMStore& dqm_store,
                                      std::string const& map_type,
                                      const edm::EventSetup& eSetup) {
  const SiStripFedCabling* fedcabling = detCabling_->fedCabling();

  if (!fedcabling)
    return;

  edm::ESHandle<TrackerTopology> tTopoHandle;
  eSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  trackerMap_ = std::make_unique<TrackerMap>(tkmapPset, fedcabling, tTopo);
  std::string tmap_title = " Tracker Map from  " + map_type;
  trackerMap_->setTitle(tmap_title);

  nDet_ = 0;
  tkMapMax_ = 0.0;
  tkMapMin_ = 0.0;
  meanToMaxFactor_ = 2.5;
  useSSQuality_ = false;
  ssqLabel_ = "";
  stripTopLevelDir_ = "SiStrip";

  if (map_type == "QTestAlarm") {
    setTkMapFromAlarm(dqm_store, eSetup);
  } else {
    trackerMap_->fill_all_blank();
    setTkMapFromHistogram(dqm_store, map_type, eSetup);
    setTkMapRange(map_type);
  }
  trackerMap_->printonline();
  trackerMap_.reset();
}

void SiStripTrackerMapCreator::createForOffline(const edm::ParameterSet& tkmapPset,
                                                DQMStore& dqm_store,
                                                std::string& map_type,
                                                const edm::EventSetup& eSetup) {
  // Determine the strip top level dirctory in the DQM file: it is the
  // path where MechanicalView is minus one directory
  std::string const mdir{"MechanicalView"};
  dqm_store.cd();
  if (!SiStripUtility::goToDir(dqm_store, mdir)) {
    edm::LogError("SiStripTopLevelDirNotFound") << "I cannot find the SiStrip top level directory in the DQM file";
  } else {
    const std::string& mechanicalview_dir = dqm_store.pwd();
    stripTopLevelDir_ = mechanicalview_dir.substr(0, mechanicalview_dir.find_last_of('/'));
    edm::LogInfo("SiStripTopLevelDirFound") << "SiStrip top level directory is " << stripTopLevelDir_;
  }
  dqm_store.cd();

  //
  const SiStripFedCabling* fedcabling = detCabling_->fedCabling();

  if (!fedcabling)
    return;

  edm::ESHandle<TrackerTopology> tTopoHandle;
  eSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  trackerMap_ = std::make_unique<TrackerMap>(tkmapPset, fedcabling, tTopo);

  meanToMaxFactor_ = tkmapPset.getUntrackedParameter<double>("meanToMaxFact", 2.5);
  useSSQuality_ = tkmapPset.getUntrackedParameter<bool>("useSSQuality", false);
  ssqLabel_ = tkmapPset.getUntrackedParameter<std::string>("ssqLabel", "");
  bool tkMapPSU = tkmapPset.getUntrackedParameter<bool>("psuMap", false);
  bool tkMapFED = tkmapPset.getUntrackedParameter<bool>("fedMap", false);
  std::string namesuffix = tkmapPset.getUntrackedParameter<std::string>("mapSuffix", "");

  //  std::string tmap_title = " Tracker Map from  " + map_type;
  unsigned int runNumber_ = tkmapPset.getUntrackedParameter<unsigned int>("RunNumber", 1);
  std::stringstream ss;
  ss << runNumber_;
  sRunNumber = ss.str();
  std::cout << sRunNumber << "\n\n\n\n\n\n";
  std::string tmap_title;
  if (runNumber_ > 0) {
    tmap_title = " Run: " + sRunNumber + ", Tracker Map from " + map_type;
  } else {
    tmap_title = " Tracker Map from " + map_type;
  }
  trackerMap_->setTitle(tmap_title);

  if (tkmapPset.exists("TopModules"))
    topModules_ = tkmapPset.getUntrackedParameter<bool>("TopModules");
  else
    topModules_ = false;

  if (tkmapPset.exists("numberTopModules"))
    numTopModules_ = tkmapPset.getUntrackedParameter<uint32_t>("numberTopModules");
  else
    numTopModules_ = 20;

  if (tkmapPset.exists("topModLabel"))
    topModLabel_ = tkmapPset.getUntrackedParameter<int32_t>("topModLabel");
  else
    topModLabel_ = " top Modules " + map_type;

  if (map_type == "QTestAlarm") {
    setTkMapFromAlarm(dqm_store, eSetup);
  } else {
    setTkMapFromHistogram(dqm_store, map_type, eSetup);
  }
  // if not overwitten by manual configuration min=0 and max= mean value * meanToMaxFactor_
  setTkMapRangeOffline();

  // check manual setting

  if (tkmapPset.exists("mapMax"))
    tkMapMax_ = tkmapPset.getUntrackedParameter<double>("mapMax");
  if (tkmapPset.exists("mapMin"))
    tkMapMin_ = tkmapPset.getUntrackedParameter<double>("mapMin");

  if (map_type == "ResidualsMean") {
    ResidualsRMS_ = false;
    setTkMapFromHistogram(dqm_store, map_type, eSetup);
    edm::LogInfo("TkMapToBeSaved") << "Ready to save TkMap " << map_type << namesuffix << " with range set to "
                                   << tkMapMin_ << " - " << tkMapMax_;
    trackerMap_->save(true, 0.0000000001, 0.005, map_type + namesuffix + ".svg");
    trackerMap_->save(true, 0.0000000001, 0.005, map_type + namesuffix + ".png", 4500, 2400);
    ResidualsRMS_ = true;
    map_type = "ResidualsRMS";
    if (runNumber_ > 0) {
      tmap_title = " Run: " + sRunNumber + ", Tracker Map from " + map_type;
    } else {
      tmap_title = " Tracker Map from " + map_type;
    }
    trackerMap_->setTitle(tmap_title);
    setTkMapFromHistogram(dqm_store, map_type, eSetup);
    edm::LogInfo("TkMapToBeSaved") << "Ready to save TkMap " << map_type << namesuffix
                                   << " with range set to 0.0 - 1.0";
    trackerMap_->save(true, 0.0, 1.0, map_type + namesuffix + ".svg");
    trackerMap_->save(true, 0.0, 1.0, map_type + namesuffix + ".png", 4500, 2400);
  } else {
    edm::LogInfo("TkMapToBeSaved") << "Ready to save TkMap " << map_type << namesuffix << " with range set to "
                                   << tkMapMin_ << " - " << tkMapMax_;
    trackerMap_->save(true, tkMapMin_, tkMapMax_, map_type + namesuffix + ".svg");
    trackerMap_->save(true, tkMapMin_, tkMapMax_, map_type + namesuffix + ".png", 4500, 2400);
  }

  if (tkMapPSU) {
    edm::LogInfo("PSUMapToBeSaved") << "Ready to save PSU TkMap " << map_type << namesuffix << " with range set to "
                                    << tkMapMin_ << " - " << tkMapMax_;
    trackerMap_->save_as_psutrackermap(true, tkMapMin_, tkMapMax_, map_type + namesuffix + "_psu.png");
  }

  if (tkMapFED) {
    edm::LogInfo("FEDMapToBeSaved") << "Ready to save FED TkMap " << map_type << namesuffix << " with range set to "
                                    << tkMapMin_ << " - " << tkMapMax_;
    trackerMap_->save_as_fedtrackermap(true, tkMapMin_, tkMapMax_, map_type + namesuffix + "_fed.png");
  }
  trackerMap_.reset();
}

//
// -- Fill Tracker Map with QTest Alarms and SiStripQuality bad modules
//
void SiStripTrackerMapCreator::setTkMapFromAlarm(DQMStore& dqm_store, const edm::EventSetup& eSetup) {
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  eSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  nDet_ = 0;
  tkMapMax_ = 0.0;
  tkMapMin_ = 0.0;

  edm::ESHandle<SiStripQuality> ssq;

  if (useSSQuality_) {
    eSetup_.get<SiStripQualityRcd>().get(ssqLabel_, ssq);
  }

  trackerMap_->fillc_all_blank();

  std::map<unsigned int, std::string> badmodmap;

  // used to avoid multiple checks on the same detid since the loop is done on the FED channels
  uint32_t detId_save = 0;
  // example of loop using SiStripDetCabling
  for (auto const& module : detCabling_->getDetCabling()) {
    uint32_t detId = module.first;
    if (detId == 0 || detId == 0xFFFFFFFF)
      continue;
    if (detId_save != detId) {
      detId_save = detId;
      bool isBad = useSSQuality_ && ssq->IsModuleBad(detId);
      paintTkMapFromAlarm(detId, tTopo, dqm_store, isBad, badmodmap);
    } else {
      edm::LogWarning("TwiceTheSameDetId")
          << "The detid " << detId << " was found already in the loop on SiStripDetCabling";
    }
  }
  printBadModuleList(badmodmap, eSetup);
}
//
//
// -- Paint Tracker Map with QTest Alarms
//
void SiStripTrackerMapCreator::paintTkMapFromAlarm(uint32_t const det_id,
                                                   TrackerTopology const* tTopo,
                                                   DQMStore& dqm_store,
                                                   bool const isBad,
                                                   std::map<unsigned int, std::string>& badmodmap) {
  std::ostringstream comment;
  uint16_t flag = 0;
  flag = getDetectorFlagAndComment(&dqm_store, det_id, tTopo, comment);

  int rval, gval, bval;
  SiStripUtility::getDetectorStatusColor(flag, rval, gval, bval);
  if (isBad) {
    rval = 255;
    gval = 255;
    bval = 0;
    comment << " PCLBadModule ";
  }
  trackerMap_->setText(det_id, comment.str());
  trackerMap_->fillc(det_id, rval, gval, bval);

  if (flag != 0 || isBad) {
    uint lay = tTopo->layer(SiStripDetId(det_id));
    std::string layer = " Layer " + std::to_string(lay);
    badmodmap[det_id] = comment.str() + layer;
  }

  detFlag_[det_id] = flag;
}

//
// --  Paint Tracker Map from TkHistoMap Histograms
void SiStripTrackerMapCreator::setTkMapFromHistogram(DQMStore& dqm_store,
                                                     std::string const& htype,
                                                     edm::EventSetup const& eSetup) {
  dqm_store.cd();

  std::string const mdir{"MechanicalView"};
  if (!SiStripUtility::goToDir(dqm_store, mdir))
    return;
  std::string mechanicalview_dir = dqm_store.pwd();

  std::vector<std::string> subdet_folder;
  subdet_folder.push_back("TIB");
  subdet_folder.push_back("TOB");
  subdet_folder.push_back("TEC/MINUS");
  subdet_folder.push_back("TEC/PLUS");
  subdet_folder.push_back("TID/MINUS");
  subdet_folder.push_back("TID/PLUS");

  nDet_ = 0;
  tkMapMax_ = 0.0;
  tkMapMin_ = 0.0;

  auto topNmodVec = topModules_ ? new std::vector<std::pair<float, uint32_t>>{} : nullptr;
  for (auto const& sd : subdet_folder) {
    std::string dname = mechanicalview_dir + "/" + sd;
    if (!dqm_store.dirExists(dname))
      continue;
    dqm_store.cd(dname);
    std::vector<std::string> layerVec = dqm_store.getSubdirs();
    for (auto const& layer : layerVec) {
      if (layer.find("BadModuleList") != std::string::npos)
        continue;
      std::vector<MonitorElement*> meVec = dqm_store.getContents(layer);
      MonitorElement* tkhmap_me{nullptr};
      std::string name;
      for (auto tkh : meVec) {
        name = tkh->getName();
        if (name.find("TkHMap") == std::string::npos)
          continue;
        if (htype == "QTestAlarm") {
          edm::LogError("ItShouldNotBeHere") << "QTestAlarm map: you should not be here!";
          tkhmap_me = tkh;
          break;
        } else if (name.find(htype) != std::string::npos) {
          tkhmap_me = tkh;
          break;
        }
      }
      if (tkhmap_me != nullptr) {
        paintTkMapFromHistogram(tkhmap_me, htype, topNmodVec);
      }
    }
    dqm_store.cd(mechanicalview_dir);
  }
  dqm_store.cd();
  if (topNmodVec)
    printTopModules(*topNmodVec, eSetup);
}

void SiStripTrackerMapCreator::paintTkMapFromHistogram(MonitorElement const* me,
                                                       std::string const& htype,
                                                       std::vector<std::pair<float, uint32_t>>* topNmodVec) {
  const std::string& name = me->getName();
  std::string lname = name.substr(name.find("TkHMap_") + 7);
  lname = lname.substr(lname.find("_T") + 1);
  for (DetId det_id : tkDetMap_->getDetsForLayer(TkDetMap::getLayerNum(lname))) {
    if (det_id.rawId() <= 0)
      continue;
    nDet_++;
    const TkLayerMap::XYbin& xyval = tkDetMap_->getXY(det_id, cachedDetId_, cachedLayer_, cachedXYbin_);
    float fval = 0.0;
    if (name.find("Residuals") != std::string::npos) {
      if (ResidualsRMS_ == true) {
        if (me->kind() == MonitorElement::Kind::TPROFILE2D) {
          TProfile2D* tp = me->getTProfile2D();
          float fval_prov =
              tp->GetBinError(xyval.ix, xyval.iy) * sqrt(tp->GetBinEntries(tp->GetBin(xyval.ix, xyval.iy)));
          fval = fval_prov;
        }
      } else {
        float fval_prov = me->getBinContent(xyval.ix, xyval.iy);
        fval = std::abs(fval_prov);
      }
    } else
      fval = me->getBinContent(xyval.ix, xyval.iy);
    if (htype == "QTestAlarm") {
      edm::LogError("ItShouldNotBeHere") << "QTestAlarm map: you should not be here!";
    } else {
      if (fval == 0.0)
        trackerMap_->fillc(det_id, 255, 255, 255);
      else {
        trackerMap_->fill_current_val(det_id, fval);
        if (topNmodVec) {
          topNmodVec->emplace_back(fval, det_id);
        }
      }
      tkMapMax_ += fval;
    }
  }
}
//
// -- Get Flag and status Comment
//
uint16_t SiStripTrackerMapCreator::getDetectorFlagAndComment(DQMStore* const dqm_store,
                                                             uint32_t const det_id,
                                                             const TrackerTopology* tTopo,
                                                             std::ostringstream& comment) {
  comment << "Module " << det_id;
  uint16_t flag = 0;

  // get FED channels corresponding to the det_id

  comment << " FEDCHs ";
  constexpr std::size_t indent_width{7ull};
  std::vector<const FedChannelConnection*> conns = detCabling_->getConnections(det_id);
  for (auto const conn : conns) {
    if (conn) {
      comment << std::setw(3) << conn->fedId() << "/" << std::setw(2) << conn->fedCh() << " ";
    } else {
      comment << std::string(indent_width, ' ');
    }
  }
  if (conns.empty()) {
    comment << std::string(indent_width * 3, ' ');
  } else if (conns.size() == 1) {
    comment << std::string(indent_width * 2, ' ');
  } else if (conns.size() == 2) {
    comment << std::string(indent_width, ' ');
  }

  if (!dqm_store)
    return flag;

  SiStripFolderOrganizer folder_organizer;
  std::string subdet_folder, badmodule_folder;

  dqm_store->cd();

  folder_organizer.setSiStripFolderName(stripTopLevelDir_);
  folder_organizer.getSubDetFolder(det_id, tTopo, subdet_folder);

  LogDebug("SearchBadModule") << det_id << " " << subdet_folder << " " << stripTopLevelDir_;

  if (dqm_store->dirExists(subdet_folder)) {
    badmodule_folder = subdet_folder + "/BadModuleList";
    LogDebug("SearchBadModule") << subdet_folder << " exists: " << badmodule_folder;
  } else {
    edm::LogError("SubDetFolderNotFound") << subdet_folder << " does not exist for detid " << det_id;
    return flag;
  }
  if (!dqm_store->dirExists(badmodule_folder)) {
    LogDebug("BadModuleFolderNotFound") << badmodule_folder << " does not exist for detid " << det_id;
    return flag;
  }
  std::ostringstream badmodule_path;
  badmodule_path << badmodule_folder << "/" << det_id;
  LogDebug("SearchBadModule") << badmodule_folder << " exists: " << badmodule_path.str();

  auto const* bad_module_me = dqm_store->get(badmodule_path.str());
  if (bad_module_me && bad_module_me->kind() == MonitorElement::Kind::INT) {
    LogDebug("SearchBadModule") << "Monitor Element found";
    flag = bad_module_me->getIntValue();
    std::string message;
    SiStripUtility::getBadModuleStatus(flag, message);
    comment << message.c_str();
  }
  return flag;
}
//
// -- create branches for root file with tracker map values by detId and fill it
//
void SiStripTrackerMapCreator::createInfoFile(std::vector<std::string> const& map_names,
                                              TTree* tkinfo_tree,
                                              DQMStore& dqm_store) {
  std::map<std::string, float> tkhmap_value;
  int qtalarm_flag = 0;
  uint32_t det_id = 0;

  if (!tkinfo_tree) {
    edm::LogError("SiStripTrackerMapCreator::createInfoFile") << "Tree not found!";
  } else {
    tkinfo_tree->Branch("DetId", &det_id, "DetId/i");
    for (auto const& mapname : map_names) {
      if (mapname == "QTestAlarm") {
        qtalarm_flag = 0;
        tkinfo_tree->Branch(mapname.c_str(), &qtalarm_flag, std::string(mapname + "/I").c_str());
      } else {
        tkhmap_value[mapname] = -1.0;
        tkinfo_tree->Branch(mapname.c_str(), &tkhmap_value[mapname], std::string(mapname + "/F").c_str());
      }
    }

    std::string dirname = "";

    std::string mdir = "MechanicalView";
    dqm_store.cd();
    if (!SiStripUtility::goToDir(dqm_store, mdir)) {
      edm::LogError("SiStripTrackerMapCreator::createInfoFile")
          << "I cannot find the SiStrip top level directory in the DQM file";
    } else {
      const std::string& mechanicalview_dir = dqm_store.pwd();
      dirname = mechanicalview_dir.substr(0, mechanicalview_dir.find_last_of('/'));
      edm::LogInfo("SiStripTrackerMapCreator::createInfoFile") << "SiStrip top level directory is " << dirname;
    }
    dqm_store.cd();

    std::vector<TkHistoMap> tkHMaps;

    uint32_t nHists = map_names.size();

    for (uint32_t ih = 0; ih < nHists; ++ih) {
      tkHMaps.emplace_back(tkDetMap_);
      if (map_names.at(ih) != "QTestAlarm") {
        std::string tkhmap_name = "TkHMap_" + map_names.at(ih);
        tkHMaps.at(ih).loadTkHistoMap(dirname, tkhmap_name, true);
      }
    }

    edm::ESHandle<GeometricDet> geomDetHandle;
    eSetup_.get<IdealGeometryRecord>().get(geomDetHandle);
    const auto detidList = TrackerGeometryUtils::getSiStripDetIds(*geomDetHandle);
    for (auto const id : detidList) {
      det_id = id;
      for (uint32_t ih = 0; ih < nHists; ++ih) {
        if (map_names.at(ih) == "QTestAlarm") {
          qtalarm_flag = getDetectorFlag(id);
        } else {
          tkhmap_value[map_names.at(ih)] = tkHMaps.at(ih).getValue(id);
        }
      }
      if (!tkinfo_tree) {
        edm::LogError("SiStripTrackerMapCreator::createInfoFile") << "Tree not found!";
      } else {
        tkinfo_tree->Fill();
      }
    }
  }
}

// DQMStore-explicit usages

void SiStripTrackerMapCreator::printBadModuleList(std::map<unsigned int, std::string> const& badmodmap,
                                                  const edm::EventSetup& eSetup) {
  // Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  eSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  bool tibDone = false, tidSide1Done = false, tidSide2Done = false, tobDone = false, tecSide1Done = false,
       tecSide2Done = false;
  constexpr unsigned int tibFirst = 369120277 - 1, tidSide1First = 402664197 - 1, tidSide2First = 402672389 - 1,
                         tobFirst = 436228134 - 1, tecSide1First = 470044965 - 1, tecSide2First = 470307109 - 1;

  int ntib = 0, ntids1 = 0, ntids2 = 0, ntob = 0, ntecs1 = 0, ntecs2 = 0;

  for (auto const& badmod : badmodmap) {
    SiStripDetId ssdetid(badmod.first);
    if (ssdetid.subDetector() == SiStripDetId::TIB)
      ntib++;
    if (ssdetid.subDetector() == SiStripDetId::TID) {
      if (tTopo->tidSide(ssdetid) == 1)
        ntids1++;
      if (tTopo->tidSide(ssdetid) == 2)
        ntids2++;
    }
    if (ssdetid.subDetector() == SiStripDetId::TOB)
      ntob++;
    if (ssdetid.subDetector() == SiStripDetId::TEC) {
      if (tTopo->tecSide(ssdetid) == 1)
        ntecs1++;
      if (tTopo->tecSide(ssdetid) == 2)
        ntecs2++;
    }
  }

  edm::LogVerbatim("BadModuleList") << "Run: " << sRunNumber << ", Number of bad modules in total:";
  edm::LogVerbatim("BadModuleList") << "--------------------------------------------------------------";
  edm::LogVerbatim("BadModuleList") << "TIB: " << ntib;
  edm::LogVerbatim("BadModuleList") << "TID/MINUS: " << ntids1;
  edm::LogVerbatim("BadModuleList") << "TID/PLUS: " << ntids2;
  edm::LogVerbatim("BadModuleList") << "TOB: " << ntob;
  edm::LogVerbatim("BadModuleList") << "TEC/MINUS: " << ntecs1;
  edm::LogVerbatim("BadModuleList") << "TEC/PLUS: " << ntecs2;
  edm::LogVerbatim("BadModuleList") << "-------------------------------";
  edm::LogVerbatim("BadModuleList");
  edm::LogVerbatim("BadModuleList") << "List of bad modules per partition:";
  edm::LogVerbatim("BadModuleList") << "----------------------------------";

  for (auto const& badmod : badmodmap) {
    if (!tibDone && badmod.first >= tibFirst) {
      tibDone = true;
      edm::LogVerbatim("BadModuleList");
      edm::LogVerbatim("BadModuleList") << "SubDetector TIB";
      edm::LogVerbatim("BadModuleList");
    }
    if (!tidSide1Done && badmod.first >= tidSide1First) {
      tidSide1Done = true;
      edm::LogVerbatim("BadModuleList");
      edm::LogVerbatim("BadModuleList") << "SubDetector TID/MINUS";
      edm::LogVerbatim("BadModuleList");
    }
    if (!tidSide2Done && badmod.first >= tidSide2First) {
      tidSide2Done = true;
      edm::LogVerbatim("BadModuleList");
      edm::LogVerbatim("BadModuleList") << "SubDetector TID/PLUS";
      edm::LogVerbatim("BadModuleList");
    }
    if (!tobDone && badmod.first >= tobFirst) {
      tobDone = true;
      edm::LogVerbatim("BadModuleList");
      edm::LogVerbatim("BadModuleList") << "SubDetector TOB";
      edm::LogVerbatim("BadModuleList");
    }
    if (!tecSide1Done && badmod.first >= tecSide1First) {
      tecSide1Done = true;
      edm::LogVerbatim("BadModuleList");
      edm::LogVerbatim("BadModuleList") << "SubDetector TEC/MINUS";
      edm::LogVerbatim("BadModuleList");
    }
    if (!tecSide2Done && badmod.first >= tecSide2First) {
      tecSide2Done = true;
      edm::LogVerbatim("BadModuleList");
      edm::LogVerbatim("BadModuleList") << "SubDetector TEC/PLUS";
      edm::LogVerbatim("BadModuleList");
    }
    edm::LogVerbatim("BadModuleList") << badmod.second;
  }
}

void SiStripTrackerMapCreator::printTopModules(std::vector<std::pair<float, uint32_t>>& topNmodVec,
                                               const edm::EventSetup& eSetup) {
  //////////////Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  eSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  if (topNmodVec.empty())
    return;

  std::sort(topNmodVec.rbegin(), topNmodVec.rend());
  if (topNmodVec.size() > numTopModules_)
    topNmodVec.resize(numTopModules_);

  edm::LogVerbatim("TopModules") << topModLabel_;
  edm::LogVerbatim("TopModules") << "------------------------------------------------------";

  for (auto const& aPair : topNmodVec) {
    uint32_t det_id = aPair.second;
    std::ostringstream comment;
    std::string subdetector;
    SiStripDetId ssdetid(aPair.second);
    if (ssdetid.subDetector() == SiStripDetId::TIB)
      subdetector = "TIB       ";
    if (ssdetid.subDetector() == SiStripDetId::TID) {
      if (tTopo->tidSide(ssdetid) == 1)
        subdetector = "TID/MINUS ";
      if (tTopo->tidSide(ssdetid) == 2)
        subdetector = "TID/PLUS  ";
    }
    if (ssdetid.subDetector() == SiStripDetId::TOB)
      subdetector = "TOB       ";
    if (ssdetid.subDetector() == SiStripDetId::TEC) {
      if (tTopo->tecSide(ssdetid) == 1)
        subdetector = "TEC/MINUS ";
      if (tTopo->tecSide(ssdetid) == 2)
        subdetector = "TEC/PLUS  ";
    }
    uint16_t flag = getDetectorFlagAndComment(nullptr, det_id, tTopo, comment);
    if (flag == 0)
      edm::LogVerbatim("TopModules") << subdetector << comment.str() << " value: " << aPair.first;
  }
  edm::LogVerbatim("TopModules") << "------------------------------------------------------";
}

//
// -- Get Tracker Map Fill Range
//
void SiStripTrackerMapCreator::setTkMapRange(std::string const& map_type) {
  tkMapMin_ = 0.0;
  if (tkMapMax_ == 0.0) {
    if (map_type.find("FractionOfBadChannels") != std::string::npos)
      tkMapMax_ = 1.0;
    else if (map_type.find("NumberOfCluster") != std::string::npos)
      tkMapMax_ = 0.01;
    else if (map_type.find("NumberOfDigi") != std::string::npos)
      tkMapMax_ = 0.6;
    else if (map_type.find("NumberOfOffTrackCluster") != std::string::npos)
      tkMapMax_ = 100.0;
    else if (map_type.find("NumberOfOnTrackCluster") != std::string::npos)
      tkMapMax_ = 50.0;
    else if (map_type.find("StoNCorrOnTrack") != std::string::npos)
      tkMapMax_ = 200.0;
  } else {
    tkMapMax_ = tkMapMax_ / nDet_ * 1.0;
    tkMapMax_ = tkMapMax_ * meanToMaxFactor_;
  }
  trackerMap_->setRange(tkMapMin_, tkMapMax_);
}

void SiStripTrackerMapCreator::setTkMapRangeOffline() {
  tkMapMin_ = 0.0;
  if (tkMapMax_ != 0.0) {
    tkMapMax_ = tkMapMax_ / (nDet_ * 1.0);
    tkMapMax_ = tkMapMax_ * meanToMaxFactor_;
  }
}
