#include "DQM/SiStripMonitorSummary/interface/SiStripBaseCondObjDQM.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "TCanvas.h"

// -----

SiStripBaseCondObjDQM::SiStripBaseCondObjDQM(const edm::EventSetup &eSetup,
                                             edm::RunNumber_t iRun,
                                             edm::ParameterSet const &hPSet,
                                             edm::ParameterSet const &fPSet)
    : eSetup_(eSetup),
      hPSet_(hPSet),
      fPSet_(fPSet),
      cacheID_memory(0),
      dqmStore_(edm::Service<DQMStore>().operator->()),
      runNumber_(iRun) {
  reader = new SiStripDetInfoFileReader(
      edm::FileInPath(std::string("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat")).fullPath());

  Mod_On_ = fPSet_.getParameter<bool>("Mod_On");
  HistoMaps_On_ = fPSet_.getParameter<bool>("HistoMaps_On");
  SummaryOnLayerLevel_On_ = fPSet_.getParameter<bool>("SummaryOnLayerLevel_On");
  SummaryOnStringLevel_On_ = fPSet_.getParameter<bool>("SummaryOnStringLevel_On");

  GrandSummary_On_ = fPSet_.getParameter<bool>("GrandSummary_On");

  CondObj_fillId_ = hPSet_.getParameter<std::string>("CondObj_fillId");
  CondObj_name_ = hPSet_.getParameter<std::string>("CondObj_name");

  // Warning message from wrong input:
  if (SummaryOnLayerLevel_On_ && SummaryOnStringLevel_On_) {
    edm::LogWarning("SiStripBaseCondObjDQM") << "[SiStripBaseCondObjDQM::SiStripBaseCondObjDQMs] PLEASE CHECK : "
                                                "String and layer level options can not be activated together"
                                             << std::endl;
  }

  // The OR of the two conditions allow to switch on this feature for all the
  // components (if the FillConditions_PSet has the TkMap_On =true) or for
  // single MEs (if the PSet for a ME has the TkMap_On =true)
  if (fPSet_.getParameter<bool>("TkMap_On") || hPSet_.getParameter<bool>("TkMap_On"))
    bookTkMap(hPSet_.getParameter<std::string>("TkMapName"));

  minValue = hPSet_.getParameter<double>("minValue");
  maxValue = hPSet_.getParameter<double>("maxValue");
}
// -----

//======================================
// -----
void SiStripBaseCondObjDQM::analysis(const edm::EventSetup &eSetup_) {
  cacheID_current = getCache(eSetup_);

  if (cacheID_memory == cacheID_current)
    return;

  getConditionObject(eSetup_);

  // The OR of the two conditions allows to switch on this feature for all the
  // components (if the FillConditions_PSet has the ActiveDetIds_On =true) or
  // for single MEs (if the PSet for a ME has the ActiveDetIds_On =true)
  if (fPSet_.getParameter<bool>("ActiveDetIds_On") || hPSet_.getParameter<bool>("ActiveDetIds_On"))
    getActiveDetIds(eSetup_);
  else
    activeDetIds = reader->getAllDetIds();

  edm::ESHandle<TrackerTopology> tTopoHandle;
  eSetup_.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology *tTopo = tTopoHandle.product();

  selectModules(activeDetIds, tTopo);

  if (Mod_On_) {
    fillModMEs(activeDetIds, eSetup_);
  }
  if (SummaryOnLayerLevel_On_ || SummaryOnStringLevel_On_) {
    fillSummaryMEs(activeDetIds, eSetup_);
  }

  if (fPSet_.getParameter<bool>("TkMap_On") || hPSet_.getParameter<bool>("TkMap_On")) {
    std::string filename = hPSet_.getParameter<std::string>("TkMapName");
    if (!filename.empty()) {
      constexpr unsigned int kSLen = 128;
      char sRun[kSLen];
      snprintf(sRun, kSLen, "_Run_%d", runNumber_);
      filename.insert(filename.find("."), sRun);

      saveTkMap(filename, minValue, maxValue);
    }
  }
}
// -----

//=====================================
// -----
void SiStripBaseCondObjDQM::analysisOnDemand(const edm::EventSetup &eSetup_,
                                             std::string requestedSubDetector,
                                             uint32_t requestedSide,
                                             uint32_t requestedLayer) {
  getConditionObject(eSetup_);
  getActiveDetIds(eSetup_);

  std::vector<uint32_t> requestedDetIds_;
  requestedDetIds_.clear();

  edm::ESHandle<TrackerTopology> tTopoHandle;
  eSetup_.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology *tTopo = tTopoHandle.product();

  if (requestedSubDetector == "TIB") {
    SiStripSubStructure::getTIBDetectors(activeDetIds, requestedDetIds_, tTopo, requestedLayer, 0, 0, 0);
  } else if (requestedSubDetector == "TID") {
    SiStripSubStructure::getTIDDetectors(activeDetIds, requestedDetIds_, tTopo, requestedSide, requestedLayer, 0, 0);
  } else if (requestedSubDetector == "TOB") {
    SiStripSubStructure::getTOBDetectors(activeDetIds, requestedDetIds_, tTopo, requestedLayer, 0, 0);
  } else if (requestedSubDetector == "TEC") {
    SiStripSubStructure::getTECDetectors(
        activeDetIds, requestedDetIds_, tTopo, requestedSide, requestedLayer, 0, 0, 0, 0);
  }

  analysisOnDemand(eSetup_, requestedDetIds_);
}
// -----

//===========================================
// -----
void SiStripBaseCondObjDQM::analysisOnDemand(const edm::EventSetup &eSetup_, uint32_t detIdOnDemand) {
  unsigned long long cacheID_current = getCache(eSetup_);

  if (cacheID_memory == cacheID_current)
    return;

  getConditionObject(eSetup_);

  std::vector<uint32_t> vdetIdsOnDemand_;
  vdetIdsOnDemand_.push_back(detIdOnDemand);  // fillModMEs needs a vector

  fillModMEs(vdetIdsOnDemand_, eSetup_);
}
// -----
//===============================================
// -----
void SiStripBaseCondObjDQM::analysisOnDemand(const edm::EventSetup &eSetup_,
                                             const std::vector<uint32_t> &detIdsOnDemand) {
  unsigned long long cacheID_current = getCache(eSetup_);

  if (cacheID_memory == cacheID_current)
    return;

  getConditionObject(eSetup_);

  fillSummaryMEs(detIdsOnDemand, eSetup_);
}
// -----
//====================================
// -----
std::vector<uint32_t> SiStripBaseCondObjDQM::getCabledModules() {
  std::vector<uint32_t> cabledDetIds_;
  eSetup_.get<SiStripDetCablingRcd>().get(detCablingHandle_);
  detCablingHandle_->addActiveDetectorsRawIds(cabledDetIds_);

  return cabledDetIds_;
}
// -----

//=========================================================
// -----

//#FIXME : very long method. please factorize it

void SiStripBaseCondObjDQM::selectModules(std::vector<uint32_t> &detIds_, const TrackerTopology *tTopo) {
  edm::LogInfo("SiStripBaseCondObjDQM") << "[SiStripBaseCondObjDQM::selectModules] input detIds_: " << detIds_.size()
                                        << std::endl;

  if (fPSet_.getParameter<bool>("restrictModules")) {
    std::vector<DetIdSelector> included_subdetsels;
    std::vector<std::string> included_subdets =
        fPSet_.getParameter<std::vector<std::string>>("ModulesToBeIncluded_DetIdSelector");
    for (std::vector<std::string>::const_iterator wsdps = included_subdets.begin(); wsdps != included_subdets.end();
         ++wsdps) {
      included_subdetsels.push_back(DetIdSelector(*wsdps));
    }

    std::vector<uint32_t> modulesToBeIncluded;
    for (std::vector<uint32_t>::const_iterator detid = detIds_.begin(); detid != detIds_.end(); ++detid) {
      for (std::vector<DetIdSelector>::const_iterator detidsel = included_subdetsels.begin();
           detidsel != included_subdetsels.end();
           ++detidsel) {
        if (detidsel->isSelected(*detid)) {
          modulesToBeIncluded.push_back(*detid);
          break;
          //	  std::cout << "detId: " << *detid << " is selected" <<
          // std::endl;
        }
      }
    }

    // -----
    // *** exclude modules ***
    std::vector<DetIdSelector> excluded_subdetsels;
    std::vector<std::string> excluded_subdets =
        fPSet_.getParameter<std::vector<std::string>>("ModulesToBeExcluded_DetIdSelector");
    for (std::vector<std::string>::const_iterator wsdps = excluded_subdets.begin(); wsdps != excluded_subdets.end();
         ++wsdps) {
      excluded_subdetsels.push_back(DetIdSelector(*wsdps));
    }

    std::vector<uint32_t> modulesToBeExcluded;
    for (std::vector<uint32_t>::const_iterator detid = detIds_.begin(); detid != detIds_.end(); ++detid) {
      for (std::vector<DetIdSelector>::const_iterator detidsel = excluded_subdetsels.begin();
           detidsel != excluded_subdetsels.end();
           ++detidsel) {
        if (detidsel->isSelected(*detid)) {
          modulesToBeExcluded.push_back(*detid);
          break;
        }
      }
    }

    ModulesToBeExcluded_ = fPSet_.getParameter<std::vector<unsigned int>>("ModulesToBeExcluded");
    ModulesToBeIncluded_ = fPSet_.getParameter<std::vector<unsigned int>>("ModulesToBeIncluded");
    SubDetectorsToBeExcluded_ = fPSet_.getParameter<std::vector<std::string>>("SubDetectorsToBeExcluded");

    // vectors to be sorted otherwise the intersection is non computed properly

    std::sort(ModulesToBeExcluded_.begin(), ModulesToBeExcluded_.end());
    std::sort(ModulesToBeIncluded_.begin(), ModulesToBeIncluded_.end());

    if (modulesToBeExcluded.empty() && modulesToBeIncluded.empty() && ModulesToBeExcluded_.empty() &&
        ModulesToBeIncluded_.empty())
      edm::LogWarning("SiStripBaseCondObjDQM") << "[SiStripBaseCondObjDQM::selectModules] PLEASE CHECK : no modules "
                                                  "to be exclude/included in your cfg"
                                               << std::endl;

    modulesToBeIncluded.insert(modulesToBeIncluded.end(), ModulesToBeIncluded_.begin(), ModulesToBeIncluded_.end());
    edm::LogInfo("SiStripBaseCondObjDQM")
        << "[SiStripBaseCondObjDQM::selectModules] modulesToBeIncluded: " << modulesToBeIncluded.size() << std::endl;
    modulesToBeExcluded.insert(modulesToBeExcluded.end(), ModulesToBeExcluded_.begin(), ModulesToBeExcluded_.end());
    edm::LogInfo("SiStripBaseCondObjDQM")
        << "[SiStripBaseCondObjDQM::selectModules] modulesToBeExcluded: " << modulesToBeExcluded.size() << std::endl;

    // apply modules selection
    if (!modulesToBeIncluded.empty()) {
      std::vector<uint32_t> tmp;
      // The intersection of two sets is formed only by the elements that are
      // present in both sets
      set_intersection(detIds_.begin(),
                       detIds_.end(),
                       modulesToBeIncluded.begin(),
                       modulesToBeIncluded.end(),
                       inserter(tmp, tmp.begin()));
      swap(detIds_, tmp);
    }

    std::sort(detIds_.begin(), detIds_.end());
    if (!modulesToBeExcluded.empty()) {
      for (std::vector<uint32_t>::const_iterator mod = modulesToBeExcluded.begin(); mod != modulesToBeExcluded.end();
           mod++) {
        std::vector<uint32_t>::iterator detid = std::lower_bound(detIds_.begin(), detIds_.end(), *mod);
        if (detid != detIds_.end())
          detIds_.erase(detid);
        detid--;
      }
    }

    // -----
    // *** restrict to a particular subdetector ***

    if (*(SubDetectorsToBeExcluded_.begin()) != "none") {
      std::vector<uint32_t> tmp;

      for (std::vector<std::string>::const_iterator modIter_ = SubDetectorsToBeExcluded_.begin();
           modIter_ != SubDetectorsToBeExcluded_.end();
           modIter_++) {
        tmp.clear();

        if (*modIter_ == "TIB") {
          SiStripSubStructure::getTIBDetectors(detIds_, tmp, tTopo, 0, 0, 0, 0);
        } else if (*modIter_ == "TOB") {
          SiStripSubStructure::getTOBDetectors(detIds_, tmp, tTopo, 0, 0, 0);
        } else if (*modIter_ == "TID") {
          SiStripSubStructure::getTIDDetectors(detIds_, tmp, tTopo, 0, 0, 0, 0);
        } else if (*modIter_ == "TEC") {
          SiStripSubStructure::getTECDetectors(detIds_, tmp, tTopo, 0, 0, 0, 0, 0, 0);
        } else {
          edm::LogWarning("SiStripBaseCondObjDQM") << "[SiStripBaseCondObjDQM::selectModules] PLEASE CHECK : no "
                                                      "correct (name) subdetector to be excluded in your cfg"
                                                   << std::endl;
        }

        std::vector<uint32_t>::iterator iterBegin_ =
            std::lower_bound(detIds_.begin(), detIds_.end(), *min_element(tmp.begin(), tmp.end()));

        std::vector<uint32_t>::iterator iterEnd_ =
            std::lower_bound(detIds_.begin(), detIds_.end(), *max_element(tmp.begin(), tmp.end()));

        for (std::vector<uint32_t>::iterator detIter_ = iterEnd_; detIter_ != iterBegin_ - 1; detIter_--) {
          detIds_.erase(detIter_);
        }

      }  // loop SubDetectorsToBeExcluded_
    }
  }
  edm::LogInfo("SiStripBaseCondObjDQM") << "[SiStripBaseCondObjDQM::selectModules] output detIds_: " << detIds_.size()
                                        << std::endl;

  // -----
  // *** fill only one Module per layer ***

  if (fPSet_.getParameter<std::string>("ModulesToBeFilled") == "onlyOneModulePerLayer") {
    std::vector<uint32_t> tmp;
    std::vector<uint32_t> layerDetIds;

    for (unsigned int i = 1; i < 5; i++) {
      tmp.clear();
      SiStripSubStructure::getTIBDetectors(detIds_, tmp, tTopo, i, 0, 0, 0);
      if (!tmp.empty()) {
        layerDetIds.push_back(*(tmp.begin()));
      }
    }
    for (unsigned int i = 1; i < 7; i++) {
      tmp.clear();
      SiStripSubStructure::getTOBDetectors(detIds_, tmp, tTopo, i, 0, 0);
      if (!tmp.empty()) {
        layerDetIds.push_back(*(tmp.begin()));
      }
    }
    for (unsigned int i = 1; i < 4; i++) {
      tmp.clear();
      SiStripSubStructure::getTIDDetectors(detIds_, tmp, tTopo, 1, i, 0, 0);
      if (!tmp.empty()) {
        layerDetIds.push_back(*(tmp.begin()));
      }
      SiStripSubStructure::getTIDDetectors(detIds_, tmp, tTopo, 2, i, 0, 0);
      if (!tmp.empty()) {
        layerDetIds.push_back(*(tmp.begin()));
      }
    }
    for (unsigned int i = 1; i < 10; i++) {
      tmp.clear();
      SiStripSubStructure::getTECDetectors(detIds_, tmp, tTopo, 1, i, 0, 0, 0, 0);
      if (!tmp.empty()) {
        layerDetIds.push_back(*(tmp.begin()));
      }
      SiStripSubStructure::getTECDetectors(detIds_, tmp, tTopo, 2, i, 0, 0, 0, 0);
      if (!tmp.empty()) {
        layerDetIds.push_back(*(tmp.begin()));
      }
    }

    detIds_.clear();
    detIds_ = layerDetIds;
  }
  // -----

}  // selectModules
// -----

//=================================================
// -----
void SiStripBaseCondObjDQM::getModMEs(ModMEs &CondObj_ME, const uint32_t &detId_, const TrackerTopology *tTopo) {
  std::map<uint32_t, ModMEs>::const_iterator ModMEsMap_iter = ModMEsMap_.find(detId_);

  if (ModMEsMap_iter != ModMEsMap_.end()) {
    CondObj_ME = ModMEsMap_iter->second;

    if ((CondObj_fillId_ == "ProfileAndCumul" || CondObj_fillId_ == "onlyProfile") && CondObj_ME.ProfileDistr) {
      CondObj_ME.ProfileDistr->Reset();
    }

    if ((CondObj_fillId_ == "ProfileAndCumul" || CondObj_fillId_ == "onlyCumul") && CondObj_ME.CumulDistr) {
      CondObj_ME.CumulDistr->Reset();
    } else {
      edm::LogWarning("SiStripBaseCondObjDQM") << "[SiStripBaseCondObjDQM::getModMEs] PLEASE CHECK : CondObj_fillId "
                                                  "option mispelled";
    }
    return;
  }

  // --> profile defined for all CondData
  if ((CondObj_fillId_ == "ProfileAndCumul" || CondObj_fillId_ == "onlyProfile")) {
    bookProfileMEs(CondObj_ME, detId_, tTopo);
  }

  // --> cumul currently only defined for noise and apvgain
  if ((CondObj_fillId_ == "ProfileAndCumul" || CondObj_fillId_ == "onlyCumul") &&
      (CondObj_name_ == "noise" || CondObj_name_ == "apvgain"))
    bookCumulMEs(CondObj_ME, detId_, tTopo);

  ModMEsMap_.insert(std::make_pair(detId_, CondObj_ME));
}
// ----

//===============================================
// -----
//%FIXME: very long method, factorize
void SiStripBaseCondObjDQM::getSummaryMEs(ModMEs &CondObj_ME, const uint32_t &detId_, const TrackerTopology *tTopo) {
  std::map<uint32_t, ModMEs>::const_iterator SummaryMEsMap_iter;

  if (CondObj_name_ == "lorentzangle" && SummaryOnStringLevel_On_) {
    SummaryMEsMap_iter = SummaryMEsMap_.find(getStringNameAndId(detId_, tTopo).second);
  } else if (CondObj_name_ == "bpcorrection" && SummaryOnStringLevel_On_) {
    SummaryMEsMap_iter = SummaryMEsMap_.find(getStringNameAndId(detId_, tTopo).second);
  } else {
    SummaryMEsMap_iter = SummaryMEsMap_.find(getLayerNameAndId(detId_, tTopo).second);
  }

  if (SummaryMEsMap_iter != SummaryMEsMap_.end()) {
    return;
  }

  // FIXME t's not good that the base class has to know about which derived
  // class shoudl exist.
  // please modify this part. implement virtual functions, esplicited in the
  // derived classes
  // --> currently only profile summary defined for all condition objects except
  // quality
  if ((CondObj_fillId_ == "ProfileAndCumul" || CondObj_fillId_ == "onlyProfile") &&
      (CondObj_name_ == "pedestal" || CondObj_name_ == "noise" || CondObj_name_ == "lowthreshold" ||
       CondObj_name_ == "highthreshold" || CondObj_name_ == "apvgain" || CondObj_name_ == "bpcorrection" ||
       CondObj_name_ == "lorentzangle")) {
    if (hPSet_.getParameter<bool>("FillSummaryProfileAtLayerLevel"))
      if (!CondObj_ME.SummaryOfProfileDistr) {
        bookSummaryProfileMEs(CondObj_ME, detId_, tTopo);
      }
  }

  // --> currently only genuine cumul LA
  if ((CondObj_fillId_ == "ProfileAndCumul" || CondObj_fillId_ == "onlyCumul") &&
      (CondObj_name_ == "lorentzangle" || CondObj_name_ == "bpcorrection" || CondObj_name_ == "noise")) {
    if (hPSet_.getParameter<bool>("FillCumulativeSummaryAtLayerLevel"))
      if (!CondObj_ME.SummaryOfCumulDistr) {
        bookSummaryCumulMEs(CondObj_ME, detId_, tTopo);
      }
  }

  // --> currently only summary as a function of detId for noise, pedestal and
  // apvgain
  if (CondObj_name_ == "noise" || CondObj_name_ == "lowthreshold" || CondObj_name_ == "highthreshold" ||
      CondObj_name_ == "apvgain" || CondObj_name_ == "pedestal" || CondObj_name_ == "quality") {
    if (hPSet_.getParameter<bool>("FillSummaryAtLayerLevel"))
      if (!CondObj_ME.SummaryDistr) {
        bookSummaryMEs(CondObj_ME, detId_, tTopo);
      }
  }

  if (CondObj_name_ == "lorentzangle" && SummaryOnStringLevel_On_) {
    // FIXME getStringNameandId takes time. not need to call it every timne. put
    // the call at the beginning of the method and caache the string
    SummaryMEsMap_.insert(std::make_pair(getStringNameAndId(detId_, tTopo).second, CondObj_ME));
  } else if (CondObj_name_ == "bpcorrection" && SummaryOnStringLevel_On_) {
    // FIXME getStringNameandId takes time. not need to call it every timne. put
    // the call at the beginning of the method and caache the string
    SummaryMEsMap_.insert(std::make_pair(getStringNameAndId(detId_, tTopo).second, CondObj_ME));
  } else {
    SummaryMEsMap_.insert(std::make_pair(getLayerNameAndId(detId_, tTopo).second, CondObj_ME));
  }
}
// ----

//====================================================
// -----
void SiStripBaseCondObjDQM::bookProfileMEs(SiStripBaseCondObjDQM::ModMEs &CondObj_ME,
                                           const uint32_t &detId_,
                                           const TrackerTopology *tTopo) {
  int hProfile_NchX = 0;
  double hProfile_LowX = 0;
  double hProfile_HighX = 0;

  std::string hProfile_description;
  hProfile_description = hPSet_.getParameter<std::string>("Profile_description");

  std::string hProfile_xTitle, hProfile_yTitle;
  hProfile_xTitle = hPSet_.getParameter<std::string>("Profile_xTitle");
  hProfile_yTitle = hPSet_.getParameter<std::string>("Profile_yTitle");

  if (CondObj_name_ != "apvgain") {
    int nStrip = reader->getNumberOfApvsAndStripLength(detId_).first * 128;

    hProfile_NchX = nStrip;
    hProfile_LowX = 0.5;
    hProfile_HighX = nStrip + 0.5;
  } else {
    int nApv = reader->getNumberOfApvsAndStripLength(detId_).first;

    hProfile_NchX = nApv;
    hProfile_LowX = 0.5;
    hProfile_HighX = nApv + 0.5;
  }

  folder_organizer.setDetectorFolder(detId_, tTopo);

  std::string hProfile_Name;
  hProfile_Name = hidmanager.createHistoId(hProfile_description, "det", detId_);

  std::string hProfile;
  hProfile = hProfile_Name;

  CondObj_ME.ProfileDistr = dqmStore_->book1D(hProfile_Name, hProfile, hProfile_NchX, hProfile_LowX, hProfile_HighX);
  CondObj_ME.ProfileDistr->setAxisTitle(hProfile_xTitle, 1);
  CondObj_ME.ProfileDistr->setAxisTitle(hProfile_yTitle, 2);
}
// -----

//=============================================
// -----
void SiStripBaseCondObjDQM::bookCumulMEs(SiStripBaseCondObjDQM::ModMEs &CondObj_ME,
                                         const uint32_t &detId_,
                                         const TrackerTopology *tTopo) {
  int hCumul_NchX = 0;
  double hCumul_LowX = 0;
  double hCumul_HighX = 0;

  std::string hCumul_description;
  hCumul_description = hPSet_.getParameter<std::string>("Cumul_description");

  std::string hCumul_xTitle, hCumul_yTitle;
  hCumul_xTitle = hPSet_.getParameter<std::string>("Cumul_xTitle");
  hCumul_yTitle = hPSet_.getParameter<std::string>("Cumul_yTitle");

  hCumul_NchX = hPSet_.getParameter<int>("Cumul_NchX");
  hCumul_LowX = hPSet_.getParameter<double>("Cumul_LowX");
  hCumul_HighX = hPSet_.getParameter<double>("Cumul_HighX");

  folder_organizer.setDetectorFolder(detId_, tTopo);

  std::string hCumul_name;
  hCumul_name = hidmanager.createHistoId(hCumul_description, "det", detId_);
  ;

  std::string hCumul_title;
  hCumul_title = hCumul_name;

  CondObj_ME.CumulDistr = dqmStore_->book1D(hCumul_name, hCumul_title, hCumul_NchX, hCumul_LowX, hCumul_HighX);
  CondObj_ME.CumulDistr->setAxisTitle(hCumul_xTitle, 1);
  CondObj_ME.CumulDistr->setAxisTitle(hCumul_yTitle, 2);
}
// ----

//===========================================
// -----
//#FIXME: same comments: factorize, and remove any reference to derived classes
void SiStripBaseCondObjDQM::bookSummaryProfileMEs(SiStripBaseCondObjDQM::ModMEs &CondObj_ME,
                                                  const uint32_t &detId_,
                                                  const TrackerTopology *tTopo) {
  std::vector<uint32_t> sameLayerDetIds_;

  int hSummaryOfProfile_NchX = 0;
  double hSummaryOfProfile_LowX = 0;
  double hSummaryOfProfile_HighX = 0;

  std::string hSummaryOfProfile_description;
  hSummaryOfProfile_description = hPSet_.getParameter<std::string>("SummaryOfProfile_description");

  std::string hSummaryOfProfile_xTitle, hSummaryOfProfile_yTitle;
  hSummaryOfProfile_xTitle = hPSet_.getParameter<std::string>("SummaryOfProfile_xTitle");
  hSummaryOfProfile_yTitle = hPSet_.getParameter<std::string>("SummaryOfProfile_yTitle");

  int hSummaryOfProfile_NchY;
  double hSummaryOfProfile_LowY, hSummaryOfProfile_HighY;
  hSummaryOfProfile_NchY = hPSet_.getParameter<int>("SummaryOfProfile_NchY");
  hSummaryOfProfile_LowY = hPSet_.getParameter<double>("SummaryOfProfile_LowY");
  hSummaryOfProfile_HighY = hPSet_.getParameter<double>("SummaryOfProfile_HighY");

  int nStrip, nApv, layerId_;

  if (CondObj_name_ == "lorentzangle" && SummaryOnStringLevel_On_) {
    layerId_ = getStringNameAndId(detId_, tTopo).second;
  } else if (CondObj_name_ == "bpcorrection" && SummaryOnStringLevel_On_) {
    layerId_ = getStringNameAndId(detId_, tTopo).second;
  } else {
    layerId_ = getLayerNameAndId(detId_, tTopo).second;
  }

  if (CondObj_name_ == "pedestal" || CondObj_name_ == "noise" || CondObj_name_ == "lowthreshold" ||
      CondObj_name_ == "highthreshold") {  // plot in strip number

    if ((layerId_ > 610 && layerId_ < 620) ||  // TID & TEC have 768 strips at maximum
        (layerId_ > 620 && layerId_ < 630) || (layerId_ > 410 && layerId_ < 414) ||
        (layerId_ > 420 && layerId_ < 424)) {
      nStrip = 768;
    } else {
      nStrip = reader->getNumberOfApvsAndStripLength(detId_).first * 128;
    }

    hSummaryOfProfile_NchX = nStrip;
    hSummaryOfProfile_LowX = 0.5;
    hSummaryOfProfile_HighX = nStrip + 0.5;

  } else if (((CondObj_name_ == "lorentzangle" || CondObj_name_ == "bpcorrection") && SummaryOnLayerLevel_On_) ||
             CondObj_name_ == "quality") {  // plot in detId-number

    // -----
    // get detIds belonging to same layer to fill X-axis with detId-number

    sameLayerDetIds_.clear();

    switch (DetId(detId_).subdetId()) {
      case StripSubdetector::TIB:
        SiStripSubStructure::getTIBDetectors(
            activeDetIds, sameLayerDetIds_, tTopo, tTopo->tibLayer(detId_), 0, 0, tTopo->tibString(detId_));
        break;
      case StripSubdetector::TID:
        SiStripSubStructure::getTIDDetectors(activeDetIds, sameLayerDetIds_, tTopo, 0, 0, 0, 0);
        break;
      case StripSubdetector::TOB:
        SiStripSubStructure::getTOBDetectors(activeDetIds, sameLayerDetIds_, tTopo, tTopo->tobLayer(detId_), 0, 0);
        break;
      case StripSubdetector::TEC:
        SiStripSubStructure::getTECDetectors(activeDetIds, sameLayerDetIds_, tTopo, 0, 0, 0, 0, 0, 0);
        break;
    }

    hSummaryOfProfile_NchX = sameLayerDetIds_.size();
    hSummaryOfProfile_LowX = 0.5;
    hSummaryOfProfile_HighX = sameLayerDetIds_.size() + 0.5;

  } else if ((CondObj_name_ == "lorentzangle" || CondObj_name_ == "bpcorrection") &&
             SummaryOnStringLevel_On_) {  // plot in detId-number

    // -----
    // get detIds belonging to same string to fill X-axis with detId-number

    sameLayerDetIds_.clear();

    switch (DetId(detId_).subdetId()) {
      case StripSubdetector::TIB:
        if (tTopo->tibIsInternalString(detId_)) {
          SiStripSubStructure::getTIBDetectors(
              activeDetIds, sameLayerDetIds_, tTopo, tTopo->tibLayer(detId_), 0, 1, tTopo->tibString(detId_));
        } else if (tTopo->tibIsExternalString(detId_)) {
          SiStripSubStructure::getTIBDetectors(
              activeDetIds, sameLayerDetIds_, tTopo, tTopo->tibLayer(detId_), 0, 2, tTopo->tibString(detId_));
        }
        break;
      case StripSubdetector::TID:
        SiStripSubStructure::getTIDDetectors(activeDetIds, sameLayerDetIds_, tTopo, 0, 0, 0, 0);
        break;
      case StripSubdetector::TOB:
        SiStripSubStructure::getTOBDetectors(
            activeDetIds, sameLayerDetIds_, tTopo, tTopo->tobLayer(detId_), 0, tTopo->tobRod(detId_));
        break;
      case StripSubdetector::TEC:
        SiStripSubStructure::getTECDetectors(activeDetIds, sameLayerDetIds_, tTopo, 0, 0, 0, 0, 0, 0);
        break;
    }

    hSummaryOfProfile_NchX = sameLayerDetIds_.size();
    hSummaryOfProfile_LowX = 0.5;
    hSummaryOfProfile_HighX = sameLayerDetIds_.size() + 0.5;

  } else if (CondObj_name_ == "apvgain") {
    if ((layerId_ > 610 && layerId_ < 620) ||  // TID & TEC have 6 apvs at maximum
        (layerId_ > 620 && layerId_ < 630) || (layerId_ > 410 && layerId_ < 414) ||
        (layerId_ > 420 && layerId_ < 424)) {
      nApv = 6;
    } else {
      nApv = reader->getNumberOfApvsAndStripLength(detId_).first;
    }

    hSummaryOfProfile_NchX = nApv;
    hSummaryOfProfile_LowX = 0.5;
    hSummaryOfProfile_HighX = nApv + 0.5;

  } else {
    edm::LogWarning("SiStripBaseCondObjDQM") << "[SiStripBaseCondObjDQM::bookSummaryProfileMEs] PLEASE CHECK : "
                                                "x-axis label in your cfg"
                                             << std::endl;
  }

  uint32_t layer_ = 0;

  layer_ = folder_organizer.GetSubDetAndLayer(detId_, tTopo).second;

  folder_organizer.setLayerFolder(detId_, tTopo, layer_);

  std::string hSummaryOfProfile_name;

  // ---
  int subdetectorId_ = ((detId_ >> 25) & 0x7);

  if (subdetectorId_ < 3 || subdetectorId_ > 6) {
    edm::LogError("SiStripBaseCondObjDQM") << "[SiStripBaseCondObjDQM::bookSummaryProfileMEs] WRONG INPUT : no "
                                              "such subdetector type : "
                                           << subdetectorId_ << " no folder set!" << std::endl;
    return;
  }
  // ---

  if ((CondObj_name_ == "lorentzangle" || CondObj_name_ == "bpcorrection") && SummaryOnStringLevel_On_) {
    hSummaryOfProfile_name = hidmanager.createHistoLayer(
        hSummaryOfProfile_description, "layer", getStringNameAndId(detId_, tTopo).first, "");
  } else {
    hSummaryOfProfile_name =
        hidmanager.createHistoLayer(hSummaryOfProfile_description, "layer", getLayerNameAndId(detId_, tTopo).first, "");
  }

  std::string hSummaryOfProfile_title;
  hSummaryOfProfile_title = hSummaryOfProfile_name;

  CondObj_ME.SummaryOfProfileDistr = dqmStore_->bookProfile(hSummaryOfProfile_name,
                                                            hSummaryOfProfile_title,
                                                            hSummaryOfProfile_NchX,
                                                            hSummaryOfProfile_LowX,
                                                            hSummaryOfProfile_HighX,
                                                            hSummaryOfProfile_NchY,
                                                            0.,
                                                            0.);
  //						            hSummaryOfProfile_LowY,
  //						            hSummaryOfProfile_HighY);
  CondObj_ME.SummaryOfProfileDistr->setAxisTitle(hSummaryOfProfile_xTitle, 1);
  CondObj_ME.SummaryOfProfileDistr->setAxisTitle(hSummaryOfProfile_yTitle, 2);
  CondObj_ME.SummaryOfProfileDistr->setAxisRange(hSummaryOfProfile_LowY, hSummaryOfProfile_HighY, 2);

  // -----
  // in order to get the right detId-number labelled in right bin of x-axis

  if (CondObj_name_ == "quality") {
    unsigned int iBin = 0;

    for (unsigned int i = 0; i < sameLayerDetIds_.size(); i++) {
      iBin++;
      char sameLayerDetIds_Name[1024];
      sprintf(sameLayerDetIds_Name, "%u", sameLayerDetIds_[i]);
      CondObj_ME.SummaryOfProfileDistr->setBinLabel(iBin, sameLayerDetIds_Name);
    }
  }
  if (CondObj_name_ == "lorentzangle" || CondObj_name_ == "bpcorrection") {
    // Put the detIds for the -z side as following the geometrical order:
    reverse(sameLayerDetIds_.begin(), sameLayerDetIds_.begin() + sameLayerDetIds_.size() / 2);

    unsigned int iBin = 0;
    for (unsigned int i = 0; i < sameLayerDetIds_.size(); i++) {
      iBin++;
      if (!SummaryOnStringLevel_On_) {
        // remove the label for detIds:
        CondObj_ME.SummaryOfProfileDistr->setBinLabel(iBin, "");
      }

      if (SummaryOnStringLevel_On_) {
        // Label with module position instead of detIds:
        char sameLayerDetIds_Name[1024];
        if (subdetectorId_ == 3) {  // re-abelling for TIB
          if (tTopo->tibIsZPlusSide(sameLayerDetIds_[i])) {
            sprintf(sameLayerDetIds_Name, "%i", tTopo->tibModule(sameLayerDetIds_[i]));
          } else if (tTopo->tibIsZMinusSide(sameLayerDetIds_[i])) {
            sprintf(sameLayerDetIds_Name, "%i", -tTopo->tibModule(sameLayerDetIds_[i]));
          }
          CondObj_ME.SummaryOfProfileDistr->setBinLabel(iBin, sameLayerDetIds_Name);
        } else if (subdetectorId_ == 5) {  // re-abelling for TOB
          if (tTopo->tobIsZPlusSide(sameLayerDetIds_[i])) {
            sprintf(sameLayerDetIds_Name, "%i", tTopo->tobModule(sameLayerDetIds_[i]));
          } else if (tTopo->tobIsZMinusSide(sameLayerDetIds_[i])) {
            sprintf(sameLayerDetIds_Name, "%i", -tTopo->tobModule(sameLayerDetIds_[i]));
          }
          CondObj_ME.SummaryOfProfileDistr->setBinLabel(iBin, sameLayerDetIds_Name);
        }
      }
    }

    // -----

  }  // if "lorentzangle"
}
// ----

//=============================================================
// -----
void SiStripBaseCondObjDQM::bookSummaryCumulMEs(SiStripBaseCondObjDQM::ModMEs &CondObj_ME,
                                                const uint32_t &detId_,
                                                const TrackerTopology *tTopo) {
  int hSummaryOfCumul_NchX = 0;
  double hSummaryOfCumul_LowX = 0;
  double hSummaryOfCumul_HighX = 0;

  std::string hSummaryOfCumul_description;
  hSummaryOfCumul_description = hPSet_.getParameter<std::string>("SummaryOfCumul_description");

  std::string hSummaryOfCumul_xTitle, hSummaryOfCumul_yTitle;
  hSummaryOfCumul_xTitle = hPSet_.getParameter<std::string>("SummaryOfCumul_xTitle");
  hSummaryOfCumul_yTitle = hPSet_.getParameter<std::string>("SummaryOfCumul_yTitle");

  hSummaryOfCumul_NchX = hPSet_.getParameter<int>("SummaryOfCumul_NchX");
  hSummaryOfCumul_LowX = hPSet_.getParameter<double>("SummaryOfCumul_LowX");
  hSummaryOfCumul_HighX = hPSet_.getParameter<double>("SummaryOfCumul_HighX");

  uint32_t layer_ = 0;

  layer_ = folder_organizer.GetSubDetAndLayer(detId_, tTopo).second;

  folder_organizer.setLayerFolder(detId_, tTopo, layer_);

  std::string hSummaryOfCumul_name;

  // ---
  int subdetectorId_ = ((detId_ >> 25) & 0x7);

  if (subdetectorId_ < 3 || subdetectorId_ > 6) {
    edm::LogError("SiStripBaseCondObjDQM") << "[SiStripBaseCondObjDQM::bookSummaryCumulMEs] WRONG INPUT : no such "
                                              "subdetector type : "
                                           << subdetectorId_ << " no folder set!" << std::endl;
    return;
  }
  // ---

  // LA and BP Histos are plotted for each string:
  if ((CondObj_name_ == "lorentzangle" || CondObj_name_ == "bpcorrection") && SummaryOnStringLevel_On_) {
    hSummaryOfCumul_name =
        hidmanager.createHistoLayer(hSummaryOfCumul_description, "layer", getStringNameAndId(detId_, tTopo).first, "");
  } else {
    hSummaryOfCumul_name =
        hidmanager.createHistoLayer(hSummaryOfCumul_description, "layer", getLayerNameAndId(detId_, tTopo).first, "");
  }

  std::string hSummaryOfCumul_title;
  hSummaryOfCumul_title = hSummaryOfCumul_name;

  CondObj_ME.SummaryOfCumulDistr = dqmStore_->book1D(
      hSummaryOfCumul_name, hSummaryOfCumul_title, hSummaryOfCumul_NchX, hSummaryOfCumul_LowX, hSummaryOfCumul_HighX);

  CondObj_ME.SummaryOfCumulDistr->setAxisTitle(hSummaryOfCumul_xTitle, 1);
  CondObj_ME.SummaryOfCumulDistr->setAxisTitle(hSummaryOfCumul_yTitle, 2);
}
// -----

//================================================
// -----
// FIXME same as before: factorize
void SiStripBaseCondObjDQM::bookSummaryMEs(SiStripBaseCondObjDQM::ModMEs &CondObj_ME,
                                           const uint32_t &detId_,
                                           const TrackerTopology *tTopo) {
  std::vector<uint32_t> sameLayerDetIds_;

  int hSummary_NchX = 0;
  double hSummary_LowX = 0;
  double hSummary_HighX = 0;

  std::string hSummary_description;
  hSummary_description = hPSet_.getParameter<std::string>("Summary_description");

  std::string hSummary_xTitle, hSummary_yTitle;
  hSummary_xTitle = hPSet_.getParameter<std::string>("Summary_xTitle");
  hSummary_yTitle = hPSet_.getParameter<std::string>("Summary_yTitle");

  int hSummary_NchY;
  double hSummary_LowY, hSummary_HighY;
  hSummary_NchY = hPSet_.getParameter<int>("Summary_NchY");
  hSummary_LowY = hPSet_.getParameter<double>("Summary_LowY");
  hSummary_HighY = hPSet_.getParameter<double>("Summary_HighY");

  // -----
  // get detIds belonging to same layer to fill X-axis with detId-number

  sameLayerDetIds_.clear();

  sameLayerDetIds_ = GetSameLayerDetId(activeDetIds, detId_, tTopo);

  hSummary_NchX = sameLayerDetIds_.size();
  hSummary_LowX = 0.5;
  hSummary_HighX = sameLayerDetIds_.size() + 0.5;

  uint32_t layer_ = 0;

  layer_ = folder_organizer.GetSubDetAndLayer(detId_, tTopo).second;

  folder_organizer.setLayerFolder(detId_, tTopo, layer_);

  std::string hSummary_name;

  // ---
  int subdetectorId_ = ((detId_ >> 25) & 0x7);

  if (subdetectorId_ < 3 || subdetectorId_ > 6) {
    edm::LogError("SiStripBaseCondObjDQM") << "[SiStripBaseCondObjDQM::bookSummaryMEs] WRONG INPUT : no such "
                                              "subdetector type : "
                                           << subdetectorId_ << " no folder set!" << std::endl;
    return;
  }
  // ---

  hSummary_name =
      hidmanager.createHistoLayer(hSummary_description, "layer", getLayerNameAndId(detId_, tTopo).first, "");

  std::string hSummary_title;
  hSummary_title = hSummary_name;

  CondObj_ME.SummaryDistr = dqmStore_->bookProfile(
      hSummary_name, hSummary_title, hSummary_NchX, hSummary_LowX, hSummary_HighX, hSummary_NchY, 0., 0.);
  //						   hSummary_LowY,
  //						   hSummary_HighY);
  CondObj_ME.SummaryDistr->setAxisTitle(hSummary_xTitle, 1);
  CondObj_ME.SummaryDistr->setAxisTitle(hSummary_yTitle, 2);
  CondObj_ME.SummaryDistr->setAxisRange(hSummary_LowY, hSummary_HighY, 2);

  // -----
  // in order to get the right detId-number labelled in right bin of x-axis
  unsigned int iBin = 0;

  for (unsigned int i = 0; i < sameLayerDetIds_.size(); i++) {
    iBin++;
    char sameLayerDetIds_Name[1024];
    sprintf(sameLayerDetIds_Name, "%u", sameLayerDetIds_[i]);
    if (iBin % 100 == 0)
      CondObj_ME.SummaryDistr->setBinLabel(iBin, sameLayerDetIds_Name);
  }
  // -----
}

//==========================================================
// -----
std::pair<std::string, uint32_t> SiStripBaseCondObjDQM::getLayerNameAndId(const uint32_t &detId_,
                                                                          const TrackerTopology *tTopo) {
  int subdetectorId_ = ((detId_ >> 25) & 0x7);
  int layerId_ = 0;

  std::stringstream layerName;

  if (subdetectorId_ == 3) {  // TIB

    for (unsigned int i = 1; i < 5; i++) {
      if (tTopo->tibLayer(detId_) == i) {
        layerName << "TIB__layer__" << i;
        layerId_ = 300 + i;
      }
    }

  }

  else if (subdetectorId_ == 4) {  // TIDD

    if (tTopo->tidSide(detId_) == 1) {  // TIDD side 1

      for (unsigned int i = 1; i < 4; i++) {
        if (tTopo->tidWheel(detId_) == i) {
          layerName << "TID__side__1__wheel__" << i;
          layerId_ = 410 + i;
        }
      }

    }

    else if (tTopo->tidSide(detId_) == 2) {  // TIDD side 2

      for (unsigned int i = 1; i < 4; i++) {
        if (tTopo->tidWheel(detId_) == i) {
          layerName << "TID__side__2__wheel__" << i;
          layerId_ = 420 + i;
        }
      }
    }

  }

  else if (subdetectorId_ == 5) {  // TOB

    for (unsigned int i = 1; i < 7; i++) {
      if (tTopo->tobLayer(detId_) == i) {
        layerName << "TOB__layer__" << i;
        layerId_ = 500 + i;
      }
    }

  }

  else if (subdetectorId_ == 6) {  // TEC

    if (tTopo->tecSide(detId_) == 1) {  // TEC side 1

      for (unsigned int i = 1; i < 10; i++) {
        if (tTopo->tecWheel(detId_) == i) {
          layerName << "TEC__side__1__wheel__" << i;
          layerId_ = 610 + i;
        }
      }

    }

    else if (tTopo->tecSide(detId_) == 2) {  // TEC side 2

      for (unsigned int i = 1; i < 10; i++) {
        if (tTopo->tecWheel(detId_) == i) {
          layerName << "TEC__side__2__wheel__" << i;
          layerId_ = 620 + i;
        }
      }
    }
  }

  return std::make_pair(layerName.str(), layerId_);
}

//=================================================
//---------------

std::pair<std::string, uint32_t> SiStripBaseCondObjDQM::getStringNameAndId(const uint32_t &detId_,
                                                                           const TrackerTopology *tTopo) {
  int subdetectorId_ = ((detId_ >> 25) & 0x7);
  int layerStringId_ = 0;

  std::stringstream layerStringName;

  if (subdetectorId_ == 3) {                                                   // TIB
    if (tTopo->tibLayer(detId_) == 1 && tTopo->tibIsInternalString(detId_)) {  // 1st layer int
      for (unsigned int i = 1; i < 27; i++) {
        if (tTopo->tibString(detId_) == i) {
          layerStringName << "TIB_L1_Int_Str_" << i;
          layerStringId_ = 30110 + i;
        }
      }
    } else if (tTopo->tibLayer(detId_) == 1 && tTopo->tibIsExternalString(detId_)) {  // 1st layer ext
      for (unsigned int i = 1; i < 31; i++) {
        if (tTopo->tibString(detId_) == i) {
          layerStringName << "TIB_L1_Ext_Str_" << i;
          layerStringId_ = 301200 + i;
        }
      }
    } else if (tTopo->tibLayer(detId_) == 2 && tTopo->tibIsInternalString(detId_)) {  // 2nd layer int
      for (unsigned int i = 1; i < 35; i++) {
        if (tTopo->tibString(detId_) == i) {
          layerStringName << "TIB_L2_Int_Str_" << i;
          layerStringId_ = 302100 + i;
        }
      }
    } else if (tTopo->tibLayer(detId_) == 2 && tTopo->tibIsExternalString(detId_)) {  // 2nd layer ext
      for (unsigned int i = 1; i < 39; i++) {
        if (tTopo->tibString(detId_) == i) {
          layerStringName << "TIB_L2_Ext_Str_" << i;
          layerStringId_ = 302200 + i;
        }
      }
    } else if (tTopo->tibLayer(detId_) == 3 && tTopo->tibIsInternalString(detId_)) {  // 3rd layer int
      for (unsigned int i = 1; i < 45; i++) {
        if (tTopo->tibString(detId_) == i) {
          layerStringName << "TIB_L3_Int_Str_" << i;
          layerStringId_ = 303100 + i;
        }
      }
    } else if (tTopo->tibLayer(detId_) == 3 && tTopo->tibIsExternalString(detId_)) {  // 3rd layer ext
      for (unsigned int i = 1; i < 47; i++) {
        if (tTopo->tibString(detId_) == i) {
          layerStringName << "TIB_L3_Ext_Str_" << i;
          layerStringId_ = 303200 + i;
        }
      }
    } else if (tTopo->tibLayer(detId_) == 4 && tTopo->tibIsInternalString(detId_)) {  // 4th layer int
      for (unsigned int i = 1; i < 53; i++) {
        if (tTopo->tibString(detId_) == i) {
          layerStringName << "TIB_L4_Int_Str_" << i;
          layerStringId_ = 304100 + i;
        }
      }
    } else if (tTopo->tibLayer(detId_) == 4 && tTopo->tibIsExternalString(detId_)) {  // 4th layer ext
      for (unsigned int i = 1; i < 57; i++) {
        if (tTopo->tibString(detId_) == i) {
          layerStringName << "TIB_L4_Ext_Str_" << i;
          layerStringId_ = 304200 + i;
        }
      }
    }
  }  // TIB

  else if (subdetectorId_ == 5) {        // TOB
    if (tTopo->tobLayer(detId_) == 1) {  // 1st layer
      for (unsigned int i = 1; i < 43; i++) {
        if (tTopo->tobRod(detId_) == i) {
          layerStringName << "TOB_L1_Rod_" << i;
          layerStringId_ = 50100 + i;
        }
      }
    } else if (tTopo->tobLayer(detId_) == 2) {  // 2nd layer
      for (unsigned int i = 1; i < 49; i++) {
        if (tTopo->tobRod(detId_) == i) {
          layerStringName << "TOB_L2_Rod_" << i;
          layerStringId_ = 50200 + i;
        }
      }
    } else if (tTopo->tobLayer(detId_) == 3) {  // 3rd layer
      for (unsigned int i = 1; i < 55; i++) {
        if (tTopo->tobRod(detId_) == i) {
          layerStringName << "TOB_L3_Rod_" << i;
          layerStringId_ = 50300 + i;
        }
      }
    } else if (tTopo->tobLayer(detId_) == 4) {  // 4th layer
      for (unsigned int i = 1; i < 61; i++) {
        if (tTopo->tobRod(detId_) == i) {
          layerStringName << "TOB_L4_Rod_" << i;
          layerStringId_ = 50400 + i;
        }
      }
    } else if (tTopo->tobLayer(detId_) == 5) {  // 5th layer
      for (unsigned int i = 1; i < 67; i++) {
        if (tTopo->tobRod(detId_) == i) {
          layerStringName << "TOB_L5_Rod_" << i;
          layerStringId_ = 50500 + i;
        }
      }
    } else if (tTopo->tobLayer(detId_) == 6) {  // 6st layer
      for (unsigned int i = 1; i < 75; i++) {
        if (tTopo->tobRod(detId_) == i) {
          layerStringName << "TOB_L6_Rod_" << i;
          layerStringId_ = 50600 + i;
        }
      }
    }
  }  // TOB

  return std::make_pair(layerStringName.str(), layerStringId_);
}

//========================
std::vector<uint32_t> SiStripBaseCondObjDQM::GetSameLayerDetId(const std::vector<uint32_t> &activeDetIds,
                                                               uint32_t selDetId,
                                                               const TrackerTopology *tTopo) {
  std::vector<uint32_t> sameLayerDetIds;
  sameLayerDetIds.clear();

  switch (DetId(selDetId).subdetId()) {
    case StripSubdetector::TIB:
      SiStripSubStructure::getTIBDetectors(activeDetIds, sameLayerDetIds, tTopo, tTopo->tibLayer(selDetId), 0, 0, 0);
      break;
    case StripSubdetector::TID:
      SiStripSubStructure::getTIDDetectors(
          activeDetIds, sameLayerDetIds, tTopo, tTopo->tidSide(selDetId), tTopo->tidWheel(selDetId), 0, 0);
      break;
    case StripSubdetector::TOB:
      SiStripSubStructure::getTOBDetectors(activeDetIds, sameLayerDetIds, tTopo, tTopo->tobLayer(selDetId), 0, 0);
      break;
    case StripSubdetector::TEC:
      SiStripSubStructure::getTECDetectors(
          activeDetIds, sameLayerDetIds, tTopo, tTopo->tecSide(selDetId), tTopo->tecWheel(selDetId), 0, 0, 0, 0);
      break;
  }

  return sameLayerDetIds;
}

//==========================
void SiStripBaseCondObjDQM::bookTkMap(const std::string &TkMapname) { tkMap = new TrackerMap(TkMapname); }

//==========================
void SiStripBaseCondObjDQM::fillTkMap(const uint32_t &detid, const float &value) { tkMap->fill(detid, value); }

//==========================
void SiStripBaseCondObjDQM::saveTkMap(const std::string &TkMapname, double minValue, double maxValue) {
  if (!tkMapScaler.empty()) {
    // check that saturation is below x%  below minValue and above minValue, and
    // in case re-arrange.
    float th = hPSet_.getParameter<double>("saturatedFraction");

    size_t imin = 0, imax = 0;
    float entries = 0;
    for (size_t i = 0; i < tkMapScaler.size(); ++i)
      entries += tkMapScaler[i];

    float min = 0;
    for (size_t i = 0; (i < tkMapScaler.size()) && (min < th); ++i) {
      min += tkMapScaler[i] / entries;
      imin = i;
    }

    float max = 0;
    // for(size_t i=tkMapScaler.size()-1;(i>=0) && (max<th);--i){ // Wrong
    // Since i is unsigned, i >= 0 is always true,
    // and the loop termination condition is never reached.
    // We offset the loop index by one to fix this.
    for (size_t j = tkMapScaler.size(); (j > 0) && (max < th); --j) {
      size_t i = j - 1;
      max += tkMapScaler[i] / entries;
      imax = i;
    }

    // reset maxValue;
    if (maxValue < imax) {
      edm::LogInfo("SiStripBaseCondObjDQM") << "Resetting TkMap maxValue from " << maxValue << " to " << imax;
      maxValue = imax;
    }
    // reset minValue;
    if (minValue > imin) {
      edm::LogInfo("SiStripBaseCondObjDQM") << "Resetting TkMap minValue from " << minValue << " to " << imin;
      minValue = imin;
    }
  }

  tkMap->save(false, minValue, maxValue, TkMapname, 4500, 2400);
  tkMap->setPalette(1);
  tkMap->showPalette(true);
}

//==========================
void SiStripBaseCondObjDQM::end() {
  edm::LogInfo("SiStripBaseCondObjDQM") << "SiStripBaseCondObjDQM::end" << std::endl;
}

//==========================
void SiStripBaseCondObjDQM::fillModMEs(const std::vector<uint32_t> &selectedDetIds, const edm::EventSetup &es) {
  // Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology *const tTopo = tTopoHandle.product();

  ModMEs CondObj_ME;

  for (std::vector<uint32_t>::const_iterator detIter_ = selectedDetIds.begin(); detIter_ != selectedDetIds.end();
       ++detIter_) {
    fillMEsForDet(CondObj_ME, *detIter_, tTopo);
  }
}

//==========================
void SiStripBaseCondObjDQM::fillSummaryMEs(const std::vector<uint32_t> &selectedDetIds, const edm::EventSetup &es) {
  // Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology *const tTopo = tTopoHandle.product();

  for (std::vector<uint32_t>::const_iterator detIter_ = selectedDetIds.begin(); detIter_ != selectedDetIds.end();
       detIter_++) {
    fillMEsForLayer(/*SummaryMEsMap_,*/ *detIter_, tTopo);
  }

  for (std::map<uint32_t, ModMEs>::iterator iter = SummaryMEsMap_.begin(); iter != SummaryMEsMap_.end(); iter++) {
    ModMEs selME;
    selME = iter->second;

    if (hPSet_.getParameter<bool>("FillSummaryProfileAtLayerLevel") &&
        fPSet_.getParameter<bool>("OutputSummaryProfileAtLayerLevelAsImage")) {
      if (CondObj_fillId_ == "onlyProfile" || CondObj_fillId_ == "ProfileAndCumul") {
        TCanvas c1("c1");
        selME.SummaryOfProfileDistr->getTProfile()->Draw();
        std::string name(selME.SummaryOfProfileDistr->getTProfile()->GetTitle());
        name += ".png";
        c1.Print(name.c_str());
      }
    }
    if (hPSet_.getParameter<bool>("FillSummaryAtLayerLevel") &&
        fPSet_.getParameter<bool>("OutputSummaryAtLayerLevelAsImage")) {
      TCanvas c1("c1");
      selME.SummaryDistr->getTH1()->Draw();
      std::string name(selME.SummaryDistr->getTH1()->GetTitle());
      name += ".png";
      c1.Print(name.c_str());
    }
    if (hPSet_.getParameter<bool>("FillCumulativeSummaryAtLayerLevel") &&
        fPSet_.getParameter<bool>("OutputCumulativeSummaryAtLayerLevelAsImage")) {
      if (CondObj_fillId_ == "onlyCumul" || CondObj_fillId_ == "ProfileAndCumul") {
        TCanvas c1("c1");
        selME.SummaryOfCumulDistr->getTH1()->Draw();
        std::string name(selME.SummaryOfCumulDistr->getTH1()->GetTitle());
        name += ".png";
        c1.Print(name.c_str());
      }
    }
  }
}
