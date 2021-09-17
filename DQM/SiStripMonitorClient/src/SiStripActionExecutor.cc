#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQM/SiStripMonitorClient/interface/SiStripLayoutParser.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

SiStripActionExecutor::SiStripActionExecutor(edm::ParameterSet const& ps) : pSet_{ps} {
  edm::LogInfo("SiStripActionExecutor") << " Creating SiStripActionExecutor "
                                        << "\n";
}

SiStripActionExecutor::~SiStripActionExecutor() {
  edm::LogInfo("SiStripActionExecutor") << " Deleting SiStripActionExecutor "
                                        << "\n";
}

bool SiStripActionExecutor::readConfiguration() {
  if (!summaryCreator_) {
    summaryCreator_ = std::make_unique<SiStripSummaryCreator>();
  }
  auto const fpath = pSet_.getUntrackedParameter<std::string>(
      "SummaryConfigPath", "DQM/SiStripMonitorClient/data/sistrip_monitorelement_config.xml");
  return summaryCreator_->readConfiguration(fpath);
}

//
// -- Read Configuration File
//
bool SiStripActionExecutor::readTkMapConfiguration(const SiStripDetCabling* detCabling,
                                                   const TkDetMap* tkDetMap,
                                                   const TrackerTopology* tTopo) {
  tkMapCreator_ = std::make_unique<SiStripTrackerMapCreator>(detCabling, tkDetMap, tTopo);
  return tkMapCreator_.get() != nullptr;
}
//
// -- Create and Fill Summary Monitor Elements
//
void SiStripActionExecutor::createSummary(DQMStore& dqm_store) {
  if (!summaryCreator_)
    return;

  dqm_store.cd();
  std::string dname = "SiStrip/MechanicalView";
  if (dqm_store.dirExists(dname)) {
    dqm_store.cd(dname);
    summaryCreator_->createSummary(dqm_store);
  }
}
//
// -- Create and Fill Summary Monitor Elements
//
void SiStripActionExecutor::createSummaryOffline(DQMStore& dqm_store) {
  if (!summaryCreator_)
    return;

  dqm_store.cd();
  std::string dname = "MechanicalView";
  if (SiStripUtility::goToDir(dqm_store, dname)) {
    summaryCreator_->createSummary(dqm_store);
  }
  dqm_store.cd();
}
//
// -- create tracker map
//
void SiStripActionExecutor::createTkMap(edm::ParameterSet const& tkmapPset,
                                        DQMStore& dqm_store,
                                        const std::string& map_type) {
  if (tkMapCreator_)
    tkMapCreator_->create(tkmapPset, dqm_store, map_type);
}
//
// -- create tracker map for offline
//
void SiStripActionExecutor::createOfflineTkMap(edm::ParameterSet const& tkmapPset,
                                               DQMStore& dqm_store,
                                               std::string& map_type,
                                               const SiStripQuality* stripQuality) {
  if (tkMapCreator_)
    tkMapCreator_->createForOffline(tkmapPset, dqm_store, map_type, stripQuality);
}
//
// -- create root file with detId info from tracker maps
//
void SiStripActionExecutor::createTkInfoFile(std::vector<std::string> map_names,
                                             TTree* tkinfo_tree,
                                             DQMStore& dqm_store,
                                             const GeometricDet* geomDet) {
  if (!tkMapCreator_)
    return;

  tkMapCreator_->createInfoFile(map_names, tkinfo_tree, dqm_store, geomDet);
}
//
// -- Create Status Monitor Elements
//
void SiStripActionExecutor::createStatus(DQMStore& dqm_store) {
  if (qualityChecker_.get() == nullptr) {
    qualityChecker_ = std::make_unique<SiStripQualityChecker>(pSet_);
  }
  qualityChecker_->bookStatus(dqm_store);
}

void SiStripActionExecutor::fillDummyStatus() { qualityChecker_->fillDummyStatus(); }

void SiStripActionExecutor::fillStatus(DQMStore& dqm_store,
                                       const SiStripDetCabling* detcabling,
                                       const TkDetMap* tkDetMap,
                                       const TrackerTopology* tTopo) {
  qualityChecker_->fillStatus(dqm_store, detcabling, tkDetMap, tTopo);
}

void SiStripActionExecutor::fillStatusAtLumi(DQMStore& dqm_store) { qualityChecker_->fillStatusAtLumi(dqm_store); }

void SiStripActionExecutor::createDummyShiftReport() {
  std::ofstream report_file;
  report_file.open("sistrip_shift_report.txt", std::ios::out);
  report_file << " Nothing to report!!" << std::endl;
  report_file.close();
}

void SiStripActionExecutor::createShiftReport(DQMStore& dqm_store) {
  // Read layout configuration
  std::string const localPath{"DQM/SiStripMonitorClient/data/sistrip_plot_layout.xml"};
  SiStripLayoutParser layout_parser;
  layout_parser.getDocument(edm::FileInPath(localPath).fullPath());

  std::map<std::string, std::vector<std::string>> layout_map;
  if (!layout_parser.getAllLayouts(layout_map))
    return;

  std::ostringstream shift_summary;
  configWriter_ = std::make_unique<SiStripConfigWriter>();
  configWriter_->init("ShiftReport");

  // Print Report Summary Content
  shift_summary << " Report Summary Content :\n"
                << " =========================" << std::endl;
  configWriter_->createElement("ReportSummary");

  MonitorElement* me{nullptr};
  std::string report_path;
  report_path = "SiStrip/EventInfo/reportSummaryContents/SiStrip_DetFraction_TECB";
  me = dqm_store.get(report_path);
  printReportSummary(me, shift_summary, "TECB");

  report_path = "SiStrip/EventInfo/reportSummaryContents/SiStrip_DetFraction_TECF";
  me = dqm_store.get(report_path);
  printReportSummary(me, shift_summary, "TECF");

  report_path = "SiStrip/EventInfo/reportSummaryContents/SiStrip_DetFraction_TIB";
  me = dqm_store.get(report_path);
  printReportSummary(me, shift_summary, "TIB");

  report_path = "SiStrip/EventInfo/reportSummaryContents/SiStrip_DetFraction_TIDB";
  me = dqm_store.get(report_path);
  printReportSummary(me, shift_summary, "TIDB");

  report_path = "SiStrip/EventInfo/reportSummaryContents/SiStrip_DetFraction_TIDF";
  me = dqm_store.get(report_path);
  printReportSummary(me, shift_summary, "TIDF");

  report_path = "SiStrip/EventInfo/reportSummaryContents/SiStrip_DetFraction_TOB";
  me = dqm_store.get(report_path);
  printReportSummary(me, shift_summary, "TOB");

  shift_summary << std::endl;
  printShiftHistoParameters(dqm_store, layout_map, shift_summary);

  std::ofstream report_file;
  report_file.open("sistrip_shift_report.txt", std::ios::out);
  report_file << shift_summary.str() << std::endl;
  report_file.close();
  configWriter_->write("sistrip_shift_report.xml");
  configWriter_.reset();
}

void SiStripActionExecutor::printReportSummary(MonitorElement* me, std::ostringstream& str_val, std::string name) {
  str_val << " " << name << "  : ";
  std::string value;
  SiStripUtility::getMEValue(me, value);
  configWriter_->createChildElement("MonitorElement", name, "value", value);
  float fvalue = atof(value.c_str());
  if (fvalue == -1.0)
    str_val << " Dummy Value " << std::endl;
  else
    str_val << fvalue << std::endl;
}

void SiStripActionExecutor::printShiftHistoParameters(DQMStore& dqm_store,
                                                      std::map<std::string, std::vector<std::string>> const& layout_map,
                                                      std::ostringstream& str_val) {
  str_val << std::endl;
  for (auto const& [set_name, path_names] : layout_map) {
    if (set_name.find("Summary") != std::string::npos)
      continue;
    configWriter_->createElement(set_name);

    str_val << " " << set_name << " : " << std::endl;
    str_val << " ====================================" << std::endl;

    str_val << std::setprecision(2);
    str_val << setiosflags(std::ios::fixed);
    for (auto const& path_name : path_names) {
      if (path_name.empty())
        continue;
      MonitorElement* me = dqm_store.get(path_name);
      std::ostringstream entry_str, mean_str, rms_str;
      entry_str << std::setprecision(2);
      entry_str << setiosflags(std::ios::fixed);
      mean_str << std::setprecision(2);
      mean_str << setiosflags(std::ios::fixed);
      rms_str << std::setprecision(2);
      rms_str << setiosflags(std::ios::fixed);
      entry_str << std::setw(7) << me->getEntries();
      mean_str << std::setw(7) << me->getMean();
      rms_str << std::setw(7) << me->getRMS();
      configWriter_->createChildElement(
          "MonitorElement", me->getName(), "entries", entry_str.str(), "mean", mean_str.str(), "rms", rms_str.str());

      if (me)
        str_val << " " << me->getName() << " : entries = " << std::setw(7) << me->getEntries()
                << " mean = " << me->getMean() << " : rms = " << me->getRMS() << '\n';
    }
    str_val << std::endl;
  }
}

//
//  -- Print List of Modules with QTest warning or Error
//
void SiStripActionExecutor::printFaultyModuleList(DQMStore& dqm_store, std::ostringstream& str_val) {
  dqm_store.cd();

  std::string mdir = "MechanicalView";
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

  int nDetsTotal = 0;
  int nDetsWithErrorTotal = 0;
  for (auto const& sd : subdet_folder) {
    std::string dname = mechanicalview_dir + "/" + sd;
    if (!dqm_store.dirExists(dname))
      continue;
    str_val << "============\n" << sd << '\n' << "============\n" << std::endl;

    dqm_store.cd(dname);
    std::vector<std::string> module_folders;
    SiStripUtility::getModuleFolderList(dqm_store, module_folders);
    int nDets = module_folders.size();
    dqm_store.cd();

    int nDetsWithError = 0;
    std::string bad_module_folder = dname + "/" + "BadModuleList";
    if (dqm_store.dirExists(bad_module_folder)) {
      auto const meVec = dqm_store.getContents(bad_module_folder);
      for (auto me : meVec) {
        nDetsWithError++;
        uint16_t flag = me->getIntValue();
        std::string message;
        SiStripUtility::getBadModuleStatus(flag, message);
        str_val << me->getName() << " flag : " << me->getIntValue() << "  " << message << std::endl;
      }
    }
    str_val << "---------------------------------------------------------------"
               "-----\n"
            << " Detectors :  Total " << nDets << " with Error " << nDetsWithError << '\n'
            << "---------------------------------------------------------------"
               "-----\n";
    nDetsTotal += nDets;
    nDetsWithErrorTotal += nDetsWithError;
  }
  dqm_store.cd();
  str_val << "--------------------------------------------------------------------\n"
          << " Total Number of Connected Detectors : " << nDetsTotal << '\n'
          << " Total Number of Detectors with Error : " << nDetsWithErrorTotal << '\n'
          << "--------------------------------------------------------------------" << std::endl;
}
