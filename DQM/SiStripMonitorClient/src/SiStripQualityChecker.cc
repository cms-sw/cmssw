#include "DQM/SiStripMonitorClient/interface/SiStripQualityChecker.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iomanip>

SiStripQualityChecker::SiStripQualityChecker(edm::ParameterSet const& ps) : pSet_{ps} {
  edm::LogInfo("SiStripQualityChecker") << " Creating SiStripQualityChecker "
                                        << "\n";

  SubDetFolderMap.emplace("TIB", "TIB");
  SubDetFolderMap.emplace("TOB", "TOB");
  SubDetFolderMap.emplace("TECF", "TEC/PLUS");
  SubDetFolderMap.emplace("TECB", "TEC/MINUS");
  SubDetFolderMap.emplace("TIDF", "TID/PLUS");
  SubDetFolderMap.emplace("TIDB", "TID/MINUS");
}

SiStripQualityChecker::~SiStripQualityChecker() {
  edm::LogInfo("SiStripQualityChecker") << " Deleting SiStripQualityChecker "
                                        << "\n";
}
//
// -- create reportSummary MEs
//
void SiStripQualityChecker::bookStatus(DQMStore& dqm_store) {
  if (bookedStripStatus_)
    return;

  dqm_store.cd();
  std::string strip_dir = "";
  SiStripUtility::getTopFolderPath(dqm_store, "SiStrip", strip_dir);
  if (strip_dir.empty())
    strip_dir = "SiStrip";

  // Non Standard Plots and should be put outside EventInfo folder

  dqm_store.setCurrentFolder(strip_dir + "/MechanicalView");

  std::string hname, htitle;
  hname = "detFractionReportMap";
  htitle = "SiStrip Report for Good Detector Fraction";
  DetFractionReportMap = dqm_store.book2D(hname, htitle, 6, 0.5, 6.5, 9, 0.5, 9.5);
  DetFractionReportMap->setAxisTitle("Sub Detector Type", 1);
  DetFractionReportMap->setAxisTitle("Layer/Disc Number", 2);

  hname = "detFractionReportMap_hasBadChan";
  htitle = "SiStrip Report for Good Detector Fraction due to bad channels";
  DetFractionReportMap_hasBadChan = dqm_store.book2D(hname, htitle, 6, 0.5, 6.5, 9, 0.5, 9.5);
  DetFractionReportMap_hasBadChan->setAxisTitle("Sub Detector Type", 1);
  DetFractionReportMap_hasBadChan->setAxisTitle("Layer/Disc Number", 2);
  hname = "detFractionReportMap_hasTooManyDigis";
  htitle = "SiStrip Report for Good Detector Fraction due to too many digis";
  DetFractionReportMap_hasTooManyDigis = dqm_store.book2D(hname, htitle, 6, 0.5, 6.5, 9, 0.5, 9.5);
  DetFractionReportMap_hasTooManyDigis->setAxisTitle("Sub Detector Type", 1);
  DetFractionReportMap_hasTooManyDigis->setAxisTitle("Layer/Disc Number", 2);
  hname = "detFractionReportMap_hasTooManyClu";
  htitle = "SiStrip Report for Good Detector Fraction due to too many clusters";
  DetFractionReportMap_hasTooManyClu = dqm_store.book2D(hname, htitle, 6, 0.5, 6.5, 9, 0.5, 9.5);
  DetFractionReportMap_hasTooManyClu->setAxisTitle("Sub Detector Type", 1);
  DetFractionReportMap_hasTooManyClu->setAxisTitle("Layer/Disc Number", 2);
  hname = "detFractionReportMap_hasExclFed";
  htitle = "SiStrip Report for Good Detector Fraction due to excluded FEDs";
  DetFractionReportMap_hasExclFed = dqm_store.book2D(hname, htitle, 6, 0.5, 6.5, 9, 0.5, 9.5);
  DetFractionReportMap_hasExclFed->setAxisTitle("Sub Detector Type", 1);
  DetFractionReportMap_hasExclFed->setAxisTitle("Layer/Disc Number", 2);
  hname = "detFractionReportMap_hasDcsErr";
  htitle = "SiStrip Report for Good Detector Fraction due to DCS errors";
  DetFractionReportMap_hasDcsErr = dqm_store.book2D(hname, htitle, 6, 0.5, 6.5, 9, 0.5, 9.5);
  DetFractionReportMap_hasDcsErr->setAxisTitle("Sub Detector Type", 1);
  DetFractionReportMap_hasDcsErr->setAxisTitle("Layer/Disc Number", 2);

  hname = "sToNReportMap";
  htitle = "SiStrip Report for Signal-to-Noise";
  SToNReportMap = dqm_store.book2D(hname, htitle, 6, 0.5, 6.5, 9, 0.5, 9.5);
  SToNReportMap->setAxisTitle("Sub Detector Type", 1);
  SToNReportMap->setAxisTitle("Layer/Disc Number", 2);

  // this is the main reportSummary 2D plot and should be in EventInfo
  dqm_store.setCurrentFolder(strip_dir + "/EventInfo");

  hname = "reportSummaryMap";
  htitle = "SiStrip Report Summary Map";
  SummaryReportMap = dqm_store.book2D(hname, htitle, 6, 0.5, 6.5, 9, 0.5, 9.5);
  SummaryReportMap->setAxisTitle("Sub Detector Type", 1);
  SummaryReportMap->setAxisTitle("Layer/Disc Number", 2);

  SummaryReportGlobal = dqm_store.bookFloat("reportSummary");
  int ibin = 0;

  dqm_store.setCurrentFolder(strip_dir + "/EventInfo/reportSummaryContents");
  for (auto const& [det, label] : SubDetFolderMap) {
    ibin++;
    DetFractionReportMap->setBinLabel(ibin, label);
    DetFractionReportMap_hasBadChan->setBinLabel(ibin, label);
    DetFractionReportMap_hasTooManyDigis->setBinLabel(ibin, label);
    DetFractionReportMap_hasTooManyClu->setBinLabel(ibin, label);
    DetFractionReportMap_hasExclFed->setBinLabel(ibin, label);
    DetFractionReportMap_hasDcsErr->setBinLabel(ibin, label);
    SToNReportMap->setBinLabel(ibin, label);
    SummaryReportMap->setBinLabel(ibin, label);

    SubDetMEs local_mes;

    if (det == "TECF")
      local_mes.detectorTag = "TEC+";
    else if (det == "TECB")
      local_mes.detectorTag = "TEC-";
    else if (det == "TIDF")
      local_mes.detectorTag = "TID+";
    else if (det == "TIDB")
      local_mes.detectorTag = "TID-";
    else
      local_mes.detectorTag = det;

    std::string me_name;
    me_name = "SiStrip_" + det;
    local_mes.SummaryFlag = dqm_store.bookFloat(me_name);

    me_name = "SiStrip_DetFraction_" + det;
    local_mes.DetFraction = dqm_store.bookFloat(me_name);

    me_name = "SiStrip_SToNFlag_" + det;
    local_mes.SToNFlag = dqm_store.bookFloat(me_name);
    SubDetMEsMap.insert(std::pair<std::string, SubDetMEs>(det, local_mes));
  }
  bookedStripStatus_ = true;
}

void SiStripQualityChecker::fillDummyStatus() {
  resetStatus();
  if (bookedStripStatus_) {
    for (auto& pr : SubDetMEsMap) {
      auto& local_mes = pr.second;
      local_mes.SummaryFlag->Fill(-1.0);
      local_mes.DetFraction->Fill(-1.0);
      local_mes.SToNFlag->Fill(-1.0);
    }

    for (int xbin = 1; xbin < SummaryReportMap->getNbinsX() + 1; xbin++) {
      for (int ybin = 1; ybin < SummaryReportMap->getNbinsY() + 1; ybin++) {
        SummaryReportMap->Fill(xbin, ybin, -1.0);
        DetFractionReportMap->Fill(xbin, ybin, -1.0);
        DetFractionReportMap_hasBadChan->Fill(xbin, ybin, -1.0);
        DetFractionReportMap_hasTooManyDigis->Fill(xbin, ybin, -1.0);
        DetFractionReportMap_hasTooManyClu->Fill(xbin, ybin, -1.0);
        DetFractionReportMap_hasExclFed->Fill(xbin, ybin, -1.0);
        DetFractionReportMap_hasDcsErr->Fill(xbin, ybin, -1.0);
        SToNReportMap->Fill(xbin, ybin, -1.0);
      }
    }
    SummaryReportGlobal->Fill(-1.0);
  }
}
//
// -- Reset Status
//
void SiStripQualityChecker::resetStatus() {
  if (bookedStripStatus_) {
    for (auto& pr : SubDetMEsMap) {
      auto& local_mes = pr.second;
      local_mes.DetFraction->Reset();
      local_mes.SummaryFlag->Reset();
      local_mes.SToNFlag->Reset();
    }
    SummaryReportMap->Reset();
    DetFractionReportMap->Reset();
    DetFractionReportMap_hasBadChan->Reset();
    DetFractionReportMap_hasTooManyDigis->Reset();
    DetFractionReportMap_hasTooManyClu->Reset();
    DetFractionReportMap_hasExclFed->Reset();
    DetFractionReportMap_hasDcsErr->Reset();
    SToNReportMap->Reset();

    SummaryReportGlobal->Reset();
  }
}
//
// -- Fill Status
//
void SiStripQualityChecker::fillStatus(DQMStore& dqm_store,
                                       const edm::ESHandle<SiStripDetCabling>& cabling,
                                       const edm::EventSetup& eSetup) {
  if (!bookedStripStatus_)
    bookStatus(dqm_store);

  edm::ESHandle<TkDetMap> tkMapHandle;
  eSetup.get<TrackerTopologyRcd>().get(tkMapHandle);
  tkDetMap_ = tkMapHandle.product();

  fillDummyStatus();
  fillDetectorStatus(dqm_store, cabling);

  int faulty_moduleflag = pSet_.getUntrackedParameter<bool>("PrintFaultyModuleList", false);
  if (faulty_moduleflag)
    fillFaultyModuleStatus(dqm_store, eSetup);
}
//
// Fill Detector Status
//
void SiStripQualityChecker::fillDetectorStatus(DQMStore& dqm_store, const edm::ESHandle<SiStripDetCabling>& cabling) {
  unsigned int xbin = 0;
  float global_flag = 0;
  dqm_store.cd();
  std::string mdir = "MechanicalView";
  if (!SiStripUtility::goToDir(dqm_store, mdir))
    return;
  std::string mechanicalview_dir = dqm_store.pwd();

  initialiseBadModuleList();
  for (auto& [det, local_mes] : SubDetMEsMap) {
    auto cPos = SubDetFolderMap.find(det);
    if (cPos == SubDetFolderMap.end())
      continue;
    std::string dname = mechanicalview_dir + "/" + cPos->second;
    if (!dqm_store.dirExists(dname))
      continue;
    dqm_store.cd(dname);
    ++xbin;
    float flag;
    fillSubDetStatus(dqm_store, cabling, local_mes, xbin, flag);
    global_flag += flag;
  }
  global_flag = global_flag / xbin * 1.0;
  if (SummaryReportGlobal)
    SummaryReportGlobal->Fill(global_flag);
  dqm_store.cd();
}
//
// -- Fill Sub detector Reports
//
void SiStripQualityChecker::fillSubDetStatus(DQMStore& dqm_store,
                                             const edm::ESHandle<SiStripDetCabling>& cabling,
                                             SubDetMEs& mes,
                                             unsigned int xbin,
                                             float& gflag) {
  int status_flag = pSet_.getUntrackedParameter<int>("GlobalStatusFilling", 1);
  if (status_flag < 1)
    return;

  auto const subDirVec = dqm_store.getSubdirs();

  unsigned int ybin = 0;
  int tot_ndet = 0;
  int tot_errdet = 0;
  float tot_ston_stat = 0;

  for (auto const& dname : subDirVec) {
    if (dname.find("BadModuleList") != std::string::npos)
      continue;
    if (dname.find("ring") != std::string::npos)
      continue;
    std::vector<MonitorElement*> meVec;

    ybin++;
    dqm_store.cd(dname);
    meVec = dqm_store.getContents(dname);
    uint16_t ndet = 100;
    int errdet = 0;
    int errdet_hasBadChan = 0;
    int errdet_hasTooManyDigis = 0;
    int errdet_hasTooManyClu = 0;
    int errdet_hasExclFed = 0;
    int errdet_hasDcsErr = 0;

    int ston_stat = 1;
    int lnum = atoi(dname.substr(dname.find_last_of("_") + 1).c_str());
    ndet = cabling->connectedNumber(mes.detectorTag, lnum);

    getModuleStatus(dqm_store,
                    meVec,
                    errdet,
                    errdet_hasBadChan,
                    errdet_hasTooManyDigis,
                    errdet_hasTooManyClu,
                    errdet_hasExclFed,
                    errdet_hasDcsErr);

    for (auto const* me : meVec) {
      if (!me)
        continue;
      std::vector<QReport*> reports = me->getQReports();

      if (reports.empty())
        continue;
      std::string name = me->getName();

      if (name.find("Summary_ClusterStoNCorr__OnTrack") != std::string::npos) {
        int istat = SiStripUtility::getMEStatus(me);
        if (reports[0]->getQTresult() == -1) {
          ston_stat = -1;
        } else {
          if (istat == dqm::qstatus::ERROR)
            ston_stat = 0;
          else if (istat == dqm::qstatus::STATUS_OK)
            ston_stat = 1;
        }
      }
    }
    if (ndet > 0) {
      float eff_fac = 1 - (errdet * 1.0 / ndet);
      float eff_fac_hasBadChan = 1 - (errdet_hasBadChan * 1.0 / ndet);
      float eff_fac_hasTooManyDigis = 1 - (errdet_hasTooManyDigis * 1.0 / ndet);
      float eff_fac_hasTooManyClu = 1 - (errdet_hasTooManyClu * 1.0 / ndet);
      float eff_fac_hasExclFed = 1 - (errdet_hasExclFed * 1.0 / ndet);
      float eff_fac_hasDcsErr = 1 - (errdet_hasDcsErr * 1.0 / ndet);
      fillStatusHistogram(SToNReportMap, xbin, ybin, ston_stat);
      fillStatusHistogram(DetFractionReportMap, xbin, ybin, eff_fac);
      fillStatusHistogram(DetFractionReportMap_hasBadChan, xbin, ybin, eff_fac_hasBadChan);
      fillStatusHistogram(DetFractionReportMap_hasTooManyDigis, xbin, ybin, eff_fac_hasTooManyDigis);
      fillStatusHistogram(DetFractionReportMap_hasTooManyClu, xbin, ybin, eff_fac_hasTooManyClu);
      fillStatusHistogram(DetFractionReportMap_hasExclFed, xbin, ybin, eff_fac_hasExclFed);
      fillStatusHistogram(DetFractionReportMap_hasDcsErr, xbin, ybin, eff_fac_hasDcsErr);
      if (ston_stat < 0)
        fillStatusHistogram(SummaryReportMap, xbin, ybin, eff_fac);
      else
        fillStatusHistogram(SummaryReportMap, xbin, ybin, ston_stat * eff_fac);

      tot_ndet += ndet;
      tot_errdet += errdet;
      tot_ston_stat += ston_stat;
    }
    dqm_store.cd(dname);
  }
  if (tot_ndet > 0) {
    float tot_eff_fac = 1 - (tot_errdet * 1.0 / tot_ndet);
    if (mes.DetFraction)
      mes.DetFraction->Fill(tot_eff_fac);
    float tot_ston_fac = tot_ston_stat / ybin;
    if (mes.SToNFlag)
      mes.SToNFlag->Fill(tot_ston_fac);
    if (tot_ston_fac < 0) {
      gflag = tot_eff_fac;
    } else {
      gflag = std::min(tot_eff_fac, tot_ston_fac);
    }
    if (mes.SummaryFlag)
      mes.SummaryFlag->Fill(gflag);
  }
}
//
// -- Print Status Report
//
void SiStripQualityChecker::printStatusReport() {
  std::ostringstream det_summary_str;
  for (std::map<std::string, SubDetMEs>::const_iterator it = SubDetMEsMap.begin(); it != SubDetMEsMap.end(); it++) {
    std::string det = it->first;
    det_summary_str << std::setprecision(4);
    det_summary_str << std::setiosflags(std::ios::fixed);

    det_summary_str << " Printing Status for " << det << " : " << std::endl;
    SubDetMEs local_mes = it->second;

    std::string sval;
    float fval1, fval2, fval3;
    fval1 = fval2 = fval3 = -1.0;

    SiStripUtility::getMEValue(local_mes.DetFraction, sval);
    if (!sval.empty())
      fval1 = atof(sval.c_str());
    SiStripUtility::getMEValue(local_mes.SToNFlag, sval);
    if (!sval.empty())
      fval2 = atof(sval.c_str());
    SiStripUtility::getMEValue(local_mes.SummaryFlag, sval);
    if (!sval.empty())
      fval3 = atof(sval.c_str());

    det_summary_str << std::setw(7) << " % of good detectors " << fval1 << " SToN Flag           " << fval2
                    << " Summary Flag        " << fval3 << std::endl;
  }
}
//
// -- Get Module Status from Layer Level Histograms
//
void SiStripQualityChecker::getModuleStatus(DQMStore& dqm_store,
                                            std::vector<MonitorElement*>& layer_mes,
                                            int& errdet,
                                            int& errdet_hasBadChan,
                                            int& errdet_hasTooManyDigis,
                                            int& errdet_hasTooManyClu,
                                            int& errdet_hasExclFed,
                                            int& errdet_hasDcsErr) {
  std::string lname;
  std::map<uint32_t, uint16_t> bad_modules;
  for (std::vector<MonitorElement*>::const_iterator it = layer_mes.begin(); it != layer_mes.end(); it++) {
    MonitorElement* me = (*it);
    if (!me)
      continue;
    std::vector<QReport*> qreports = me->getQReports();
    if (qreports.empty())
      continue;
    std::string name = me->getName();
    std::vector<DQMChannel> bad_channels_me;
    if (me->kind() == MonitorElement::Kind::TPROFILE) {
      bad_channels_me = qreports[0]->getBadChannels();
      lname = "";
    } else if (me->kind() == MonitorElement::Kind::TPROFILE2D && name.find("TkHMap") != std::string::npos) {
      bad_channels_me = qreports[0]->getBadChannels();
      lname = name.substr(name.find("TkHMap_") + 7);
      lname = lname.substr(lname.find("_T") + 1);
    }
    for (std::vector<DQMChannel>::iterator it = bad_channels_me.begin(); it != bad_channels_me.end(); it++) {
      int xval = (*it).getBinX();
      int yval = (*it).getBinY();
      uint32_t detId = tkDetMap_->getDetFromBin(lname, xval, yval);
      std::map<uint32_t, uint16_t>::iterator iPos = bad_modules.find(detId);
      uint16_t flag;
      if (iPos != bad_modules.end()) {
        flag = iPos->second;
        SiStripUtility::setBadModuleFlag(name, flag);
        iPos->second = flag;
      } else {
        //
        // if not in the local bad module list, check the BadModuleList dir
        //
        std::ostringstream detid_str;
        detid_str << detId;
        // now in the layer/wheel dir
        const std::string& currentdir = dqm_store.pwd();
        std::string thisMEpath = currentdir.substr(0, currentdir.rfind("/")) + "/BadModuleList/" + detid_str.str();

        MonitorElement* meBadModule = dqm_store.get(thisMEpath);
        if (meBadModule) {
          std::string val_str;
          SiStripUtility::getMEValue(meBadModule, val_str);
          flag = atoi(val_str.c_str());
        } else
          flag = 0;

        SiStripUtility::setBadModuleFlag(name, flag);
        bad_modules.insert(std::pair<uint32_t, uint16_t>(detId, flag));
      }
    }
  }
  for (std::map<uint32_t, uint16_t>::const_iterator it = bad_modules.begin(); it != bad_modules.end(); it++) {
    uint32_t detId = it->first;
    uint16_t flag = it->second;
    if (((flag >> 0) & 0x1) > 0)
      errdet_hasBadChan++;
    if (((flag >> 1) & 0x1) > 0)
      errdet_hasTooManyDigis++;
    if (((flag >> 2) & 0x1) > 0)
      errdet_hasTooManyClu++;
    if (((flag >> 3) & 0x1) > 0)
      errdet_hasExclFed++;
    if (((flag >> 4) & 0x1) > 0)
      errdet_hasDcsErr++;
    std::map<uint32_t, uint16_t>::iterator iPos = badModuleList.find(detId);
    if (iPos != badModuleList.end()) {
      iPos->second = flag;
    } else {
      badModuleList.insert(std::pair<uint32_t, uint16_t>(detId, flag));
    }
  }
  errdet = bad_modules.size();
}
//
// -- Fill Report Summary Map
//
void SiStripQualityChecker::fillStatusHistogram(MonitorElement const* me,
                                                int const xbin,
                                                int const ybin,
                                                float const val) {
  if (me && me->kind() == MonitorElement::Kind::TH2F) {
    auto th2d = me->getTH2F();
    th2d->SetBinContent(xbin, ybin, val);
  }
}
//
// -- Create Monitor Elements for Modules
//
void SiStripQualityChecker::fillFaultyModuleStatus(DQMStore& dqm_store, const edm::EventSetup& eSetup) {
  if (badModuleList.empty())
    return;

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  eSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  dqm_store.cd();
  std::string mdir = "MechanicalView";
  if (!SiStripUtility::goToDir(dqm_store, mdir))
    return;
  std::string mechanical_dir = dqm_store.pwd();

  SiStripFolderOrganizer folder_organizer;
  for (auto const [detId, value] : badModuleList) {
    std::string subdet_folder;
    folder_organizer.getSubDetFolder(detId, tTopo, subdet_folder);
    if (!dqm_store.dirExists(subdet_folder)) {
      subdet_folder = mechanical_dir + subdet_folder.substr(subdet_folder.find("MechanicalView") + 14);
      if (!dqm_store.dirExists(subdet_folder))
        continue;
    }
    std::string bad_module_folder = subdet_folder + "/" + "BadModuleList";
    dqm_store.setCurrentFolder(bad_module_folder);

    std::ostringstream detid_str;
    detid_str << detId;
    std::string full_path = bad_module_folder + "/" + detid_str.str();
    MonitorElement* me = dqm_store.get(full_path);
    if (me)
      me->Reset();
    else
      me = dqm_store.bookInt(detid_str.str());
    me->Fill(value);
  }
  dqm_store.cd();
}

void SiStripQualityChecker::initialiseBadModuleList() {
  for (auto& badMod : badModuleList) {
    badMod.second = 0;
  }
}

//
// -- Fill Status information and the lumi block
//
void SiStripQualityChecker::fillStatusAtLumi(DQMStore& dqm_store) {
  if (!bookedStripStatus_)
    bookStatus(dqm_store);
  fillDummyStatus();
  fillDetectorStatusAtLumi(dqm_store);
}
//
// Fill Detector Status MEs at the Lumi block
//
void SiStripQualityChecker::fillDetectorStatusAtLumi(DQMStore& dqm_store) {
  dqm_store.cd();
  std::string rdir = "ReadoutView";
  if (!SiStripUtility::goToDir(dqm_store, rdir))
    return;
  std::string fullpath = dqm_store.pwd() + "/PerLumiSection/" + "lumiErrorFraction";
  MonitorElement* me = dqm_store.get(fullpath);
  if (me && me->kind() == MonitorElement::Kind::TH1F) {
    TH1F* th1 = me->getTH1F();
    float global_fraction = 0.0;
    float dets = 0.0;
    for (int ibin = 1; ibin <= th1->GetNbinsX(); ibin++) {
      std::string label = th1->GetXaxis()->GetBinLabel(ibin);
      std::map<std::string, SubDetMEs>::iterator iPos = SubDetMEsMap.find(label);
      if (iPos != SubDetMEsMap.end()) {
        float fraction = 1.0 - th1->GetBinContent(ibin);
        global_fraction += fraction;
        dets++;
        iPos->second.DetFraction->Fill(fraction);
        iPos->second.SToNFlag->Fill(-1.0);
        iPos->second.SummaryFlag->Fill(fraction);
      }
    }
    global_fraction = global_fraction / dets;
    if (SummaryReportGlobal)
      SummaryReportGlobal->Fill(global_fraction);
  }
  dqm_store.cd();
}
