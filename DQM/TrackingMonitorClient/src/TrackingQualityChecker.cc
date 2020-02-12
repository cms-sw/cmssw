#include "DQM/TrackingMonitorClient/interface/TrackingQualityChecker.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "DQM/TrackingMonitorClient/interface/TrackingUtility.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iomanip>
//
// -- Constructor
//
TrackingQualityChecker::TrackingQualityChecker(edm::ParameterSet const& ps)
    : pSet_(ps), verbose_(pSet_.getUntrackedParameter<bool>("verbose", false)) {
  edm::LogInfo("TrackingQualityChecker") << " Creating TrackingQualityChecker "
                                         << "\n";

  bookedTrackingGlobalStatus_ = false;
  bookedTrackingLSStatus_ = false;

  TopFolderName_ = pSet_.getUntrackedParameter<std::string>("TopFolderName", "Tracking");

  TrackingMEs tracking_mes;
  std::vector<edm::ParameterSet> TrackingGlobalQualityMEs =
      pSet_.getParameter<std::vector<edm::ParameterSet> >("TrackingGlobalQualityPSets");
  for (auto meQTset : TrackingGlobalQualityMEs) {
    std::string QTname = meQTset.getParameter<std::string>("QT");
    tracking_mes.HistoDir = meQTset.getParameter<std::string>("dir");
    tracking_mes.HistoName = meQTset.getParameter<std::string>("name");
    if (verbose_)
      std::cout << "[TrackingQualityChecker::TrackingQualityChecker] inserting " << QTname << " in TrackingMEsMap"
                << std::endl;
    TrackingMEsMap.insert(std::pair<std::string, TrackingMEs>(QTname, tracking_mes));
  }
  if (verbose_)
    std::cout << "[TrackingQualityChecker::TrackingQualityChecker] created TrackingMEsMap" << std::endl;

  TrackingLSMEs tracking_ls_mes;
  std::vector<edm::ParameterSet> TrackingLSQualityMEs =
      pSet_.getParameter<std::vector<edm::ParameterSet> >("TrackingLSQualityPSets");
  for (auto meQTset : TrackingLSQualityMEs) {
    std::string QTname = meQTset.getParameter<std::string>("QT");
    tracking_ls_mes.HistoLSDir = meQTset.exists("LSdir") ? meQTset.getParameter<std::string>("LSdir") : "";
    tracking_ls_mes.HistoLSName = meQTset.exists("LSname") ? meQTset.getParameter<std::string>("LSname") : "";
    tracking_ls_mes.HistoLSLowerCut = meQTset.exists("LSlowerCut") ? meQTset.getParameter<double>("LSlowerCut") : -1.;
    tracking_ls_mes.HistoLSUpperCut = meQTset.exists("LSupperCut") ? meQTset.getParameter<double>("LSupperCut") : -1.;
    tracking_ls_mes.TrackingFlag = nullptr;

    if (verbose_)
      std::cout << "[TrackingQualityChecker::TrackingQualityChecker] inserting " << QTname << " in TrackingMEsMap"
                << std::endl;
    TrackingLSMEsMap.insert(std::pair<std::string, TrackingLSMEs>(QTname, tracking_ls_mes));
  }
  if (verbose_)
    std::cout << "[TrackingQualityChecker::TrackingQualityChecker] created TrackingLSMEsMap" << std::endl;
}
//
// --  Destructor
//
TrackingQualityChecker::~TrackingQualityChecker() {
  edm::LogInfo("TrackingQualityChecker") << " Deleting TrackingQualityChecker "
                                         << "\n";
}
//
// -- create reportSummary MEs
//
void TrackingQualityChecker::bookGlobalStatus(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  if (verbose_)
    std::cout << "[TrackingQualityChecker::bookGlobalStatus] already booked ? "
              << (bookedTrackingGlobalStatus_ ? "yes" : "nope") << std::endl;

  if (!bookedTrackingGlobalStatus_) {
    ibooker.cd();
    edm::LogInfo("TrackingQualityChecker") << " booking TrackingQualityStatus"
                                           << "\n";

    std::string tracking_dir = "";
    TrackingUtility::getTopFolderPath(ibooker, igetter, TopFolderName_, tracking_dir);
    ibooker.setCurrentFolder(TopFolderName_ + "/EventInfo");

    TrackGlobalSummaryReportGlobal = ibooker.bookFloat("reportSummary");

    std::string hname, htitle;
    hname = "reportSummaryMap";
    htitle = "Tracking Report Summary Map";

    size_t nQT = TrackingMEsMap.size();
    if (verbose_)
      std::cout << "[TrackingQualityChecker::bookGlobalStatus] nQT: " << nQT << std::endl;
    TrackGlobalSummaryReportMap = ibooker.book2D(hname, htitle, nQT, 0.5, float(nQT) + 0.5, 1, 0.5, 1.5);
    TrackGlobalSummaryReportMap->setAxisTitle("Track Quality Type", 1);
    TrackGlobalSummaryReportMap->setAxisTitle("QTest Flag", 2);
    size_t ibin = 0;
    for (auto meQTset : TrackingMEsMap) {
      TrackGlobalSummaryReportMap->setBinLabel(ibin + 1, meQTset.first);
      ibin++;
    }

    ibooker.setCurrentFolder(TopFolderName_ + "/EventInfo/reportSummaryContents");

    for (std::map<std::string, TrackingMEs>::iterator it = TrackingMEsMap.begin(); it != TrackingMEsMap.end(); it++) {
      std::string meQTname = it->first;
      it->second.TrackingFlag = ibooker.bookFloat("Track" + meQTname);
      if (verbose_)
        std::cout << "[TrackingQualityChecker::bookGlobalStatus] " << it->first << " exists ? "
                  << it->second.TrackingFlag << std::endl;
      if (verbose_)
        std::cout << "[TrackingQualityChecker::bookGlobalStatus] DONE w/ TrackingMEsMap" << std::endl;
    }

    bookedTrackingGlobalStatus_ = true;
    ibooker.cd();
  }
}

void TrackingQualityChecker::bookLSStatus(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  if (verbose_)
    std::cout << "[TrackingQualityChecker::bookLSStatus] already booked ? "
              << (bookedTrackingLSStatus_ ? "yes" : "nope") << std::endl;

  if (!bookedTrackingLSStatus_) {
    ibooker.cd();
    edm::LogInfo("TrackingQualityChecker") << " booking TrackingQualityStatus"
                                           << "\n";

    std::string tracking_dir = "";
    TrackingUtility::getTopFolderPath(ibooker, igetter, TopFolderName_, tracking_dir);
    ibooker.setCurrentFolder(TopFolderName_ + "/EventInfo");

    TrackLSSummaryReportGlobal = ibooker.bookFloat("reportSummary");

    std::string hname, htitle;
    hname = "reportSummaryMap";
    htitle = "Tracking Report Summary Map";

    if (verbose_) {
      size_t nQT = TrackingLSMEsMap.size();
      std::cout << "[TrackingQualityChecker::bookLSStatus] nQT: " << nQT << std::endl;
    }

    ibooker.setCurrentFolder(TopFolderName_ + "/EventInfo/reportSummaryContents");
    for (std::map<std::string, TrackingLSMEs>::iterator it = TrackingLSMEsMap.begin(); it != TrackingLSMEsMap.end();
         it++) {
      std::string meQTname = it->first;
      it->second.TrackingFlag = ibooker.bookFloat("Track" + meQTname);
      if (verbose_)
        std::cout << "[TrackingQualityChecker::bookLSStatus] " << it->first << " exists ? " << it->second.TrackingFlag
                  << std::endl;
      if (verbose_)
        std::cout << "[TrackingQualityChecker::bookLSStatus] DONE w/ TrackingLSMEsMap" << std::endl;
    }

    bookedTrackingLSStatus_ = true;
    ibooker.cd();
  }
}

//
// -- Fill Dummy  Status
//
void TrackingQualityChecker::fillDummyGlobalStatus() {
  if (verbose_)
    std::cout << "[TrackingQualityChecker::fillDummyGlobalStatus] starting ..." << std::endl;

  resetGlobalStatus();
  if (verbose_)
    std::cout << "[TrackingQualityChecker::fillDummyGlobalStatus] already booked ? "
              << (bookedTrackingGlobalStatus_ ? "yes" : "nope") << std::endl;
  if (bookedTrackingGlobalStatus_) {
    TrackGlobalSummaryReportGlobal->Fill(-1.0);

    for (int ibin = 1; ibin < TrackGlobalSummaryReportMap->getNbinsX() + 1; ibin++) {
      fillStatusHistogram(TrackGlobalSummaryReportMap, ibin, 1, -1.0);
    }

    for (std::map<std::string, TrackingMEs>::iterator it = TrackingMEsMap.begin(); it != TrackingMEsMap.end(); it++)
      it->second.TrackingFlag->Fill(-1.0);
    if (verbose_)
      std::cout << "[TrackingQualityChecker::fillDummyGlobalStatus] DONE w/ TrackingMEsMap" << std::endl;
  }
}
void TrackingQualityChecker::fillDummyLSStatus() {
  if (verbose_)
    std::cout << "[TrackingQualityChecker::fillDummyLSStatus] starting ..." << std::endl;

  resetLSStatus();
  if (verbose_)
    std::cout << "[TrackingQualityChecker::fillDummyLSStatus] already booked ? "
              << (bookedTrackingLSStatus_ ? "yes" : "nope") << std::endl;
  if (bookedTrackingLSStatus_) {
    TrackLSSummaryReportGlobal->Fill(-1.0);
    for (std::map<std::string, TrackingLSMEs>::iterator it = TrackingLSMEsMap.begin(); it != TrackingLSMEsMap.end();
         it++)
      it->second.TrackingFlag->Fill(-1.0);
    if (verbose_)
      std::cout << "[TrackingQualityChecker::fillDummyLSStatus] DONE w/ TrackingLSMEsMap" << std::endl;
  }
}

//
// -- Reset Status
//
void TrackingQualityChecker::resetGlobalStatus() {
  if (verbose_)
    std::cout << "[TrackingQualityChecker::resetGlobalStatus] already booked ? "
              << (bookedTrackingGlobalStatus_ ? "yes" : "nope") << std::endl;
  if (bookedTrackingGlobalStatus_) {
    TrackGlobalSummaryReportGlobal->Reset();
    TrackGlobalSummaryReportMap->Reset();

    for (std::map<std::string, TrackingMEs>::iterator it = TrackingMEsMap.begin(); it != TrackingMEsMap.end(); it++) {
      MonitorElement* me = it->second.TrackingFlag;
      if (verbose_)
        std::cout << "[TrackingQualityChecker::resetGlobalStatus] " << it->second.HistoName << " exist ? "
                  << (it->second.TrackingFlag == nullptr ? "nope" : "yes") << " ---> " << me << std::endl;
      me->Reset();
    }
    if (verbose_)
      std::cout << "[TrackingQualityChecker::resetGlobalStatus] DONE w/ TrackingMEsMap" << std::endl;
  }
}
void TrackingQualityChecker::resetLSStatus() {
  if (verbose_)
    std::cout << "[TrackingQualityChecker::resetLSStatus] already booked ? "
              << (bookedTrackingLSStatus_ ? "yes" : "nope") << std::endl;
  if (bookedTrackingLSStatus_) {
    TrackLSSummaryReportGlobal->Reset();
    for (std::map<std::string, TrackingLSMEs>::iterator it = TrackingLSMEsMap.begin(); it != TrackingLSMEsMap.end();
         it++) {
      MonitorElement* me = it->second.TrackingFlag;
      if (verbose_)
        std::cout << "[TrackingQualityChecker::resetLSStatus] " << it->second.HistoLSName << " exist ? "
                  << (it->second.TrackingFlag == nullptr ? "nope" : "yes") << " ---> " << me << std::endl;
      me->Reset();
    }
    if (verbose_)
      std::cout << "[TrackingQualityChecker::resetLSStatus] DONE w/ TrackingLSMEsMap" << std::endl;
  }
}

//
// -- Fill Status
//
void TrackingQualityChecker::fillGlobalStatus(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  if (verbose_)
    std::cout << "[TrackingQualityChecker::fillGlobalStatus] already booked ? "
              << (bookedTrackingGlobalStatus_ ? "yes" : "nope") << std::endl;
  if (!bookedTrackingGlobalStatus_)
    bookGlobalStatus(ibooker, igetter);

  fillDummyGlobalStatus();
  fillTrackingStatus(ibooker, igetter);
  if (verbose_)
    std::cout << "[TrackingQualityChecker::fillGlobalStatus] DONE" << std::endl;
  ibooker.cd();
}

void TrackingQualityChecker::fillLSStatus(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  if (verbose_)
    std::cout << "[TrackingQualityChecker::fillLSStatus] already booked ? "
              << (bookedTrackingLSStatus_ ? "yes" : "nope") << std::endl;
  if (!bookedTrackingLSStatus_)
    bookLSStatus(ibooker, igetter);

  fillDummyLSStatus();
  fillTrackingStatusAtLumi(ibooker, igetter);
  if (verbose_)
    std::cout << "[TrackingQualityChecker::fillLSStatus] DONE" << std::endl;
  ibooker.cd();
}

//
// -- Fill Tracking Status
//
void TrackingQualityChecker::fillTrackingStatus(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  float gstatus = 0.0;

  ibooker.cd();
  if (!TrackingUtility::goToDir(ibooker, igetter, TopFolderName_))
    return;

  int ibin = 0;
  for (std::map<std::string, TrackingMEs>::iterator it = TrackingMEsMap.begin(); it != TrackingMEsMap.end(); it++) {
    if (verbose_)
      std::cout << "[TrackingQualityChecker::fillTrackingStatus] ME: " << it->first << " ["
                << it->second.TrackingFlag->getFullname() << "] flag: " << it->second.TrackingFlag->getFloatValue()
                << std::endl;

    ibin++;

    std::string localMEdirpath = it->second.HistoDir;
    std::string MEname = it->second.HistoName;

    std::vector<MonitorElement*> tmpMEvec = igetter.getContents(ibooker.pwd() + "/" + localMEdirpath);
    if (verbose_)
      std::cout << "[TrackingQualityChecker::fillTrackingStatus] tmpMEvec: " << tmpMEvec.size() << std::endl;
    MonitorElement* me = nullptr;

    size_t nMEs = 0;
    for (auto ime : tmpMEvec) {
      std::string name = ime->getName();
      if (verbose_)
        std::cout << "name: " << name << " <-- --> " << MEname << std::endl;
      if (name.find(MEname) != std::string::npos) {
        me = ime;
        nMEs++;
      }
    }
    // only one ME found
    if (verbose_)
      std::cout << "[TrackingQualityChecker::fillTrackingStatus] nMEs: " << nMEs << std::endl;
    if (nMEs == 1) {
      float status = 0.;
      for (auto ime : tmpMEvec) {
        std::string name = ime->getName();
        if (verbose_)
          std::cout << "name: " << name << " [" << MEname << "]" << std::endl;
        if (name.find(MEname) != std::string::npos) {
          me = ime;
          if (verbose_)
            std::cout << "inside the loop nQTme: " << me->getQReports().size() << "[" << ime->getFullname() << "]"
                      << std::endl;
        }
      }
      if (verbose_)
        std::cout << "me: " << me << "[" << me->getName() << ", " << me->getFullname() << "]" << std::endl;
      if (!me)
        continue;
      if (verbose_)
        std::cout << "[TrackingQualityChecker::fillTrackingStatus] status: " << status << std::endl;
      std::vector<QReport*> qt_reports = me->getQReports();
      size_t nQTme = qt_reports.size();
      if (verbose_)
        std::cout << "nQTme: " << nQTme << std::endl;
      if (nQTme != 0) {
        if (verbose_)
          std::cout << "[TrackingQualityChecker::fillTrackingStatus] qt_reports: " << qt_reports.size() << std::endl;
        // loop on possible QTs
        for (auto iQT : qt_reports) {
          status += iQT->getQTresult();
          if (verbose_)
            std::cout << "[TrackingQualityChecker::fillTrackingStatus] iQT: " << iQT->getQRName() << std::endl;
          if (verbose_)
            std::cout << "[TrackingQualityChecker::fillTrackingStatus] MEname: " << MEname
                      << " status: " << iQT->getQTresult() << " exists ? " << (it->second.TrackingFlag ? "yes " : "no ")
                      << it->second.TrackingFlag << std::endl;
          if (verbose_)
            std::cout << "[TrackingQualityChecker::fillTrackingStatus] iQT message: " << iQT->getMessage() << std::endl;
          if (verbose_)
            std::cout << "[TrackingQualityChecker::fillTrackingStatus] iQT status: " << iQT->getStatus() << std::endl;
        }
        status = status / float(nQTme);
        if (verbose_)
          std::cout << "[TrackingQualityChecker::fillTrackingStatus] MEname: " << MEname << " status: " << status
                    << std::endl;
        it->second.TrackingFlag->Fill(status);
        if (verbose_)
          std::cout << "[TrackingQualityChecker::fillTrackingStatus] TrackGlobalSummaryReportMap: "
                    << TrackGlobalSummaryReportMap << std::endl;
        fillStatusHistogram(TrackGlobalSummaryReportMap, ibin, 1, status);
      }

      if (verbose_)
        std::cout << "[TrackingQualityChecker::fillTrackingStatus] gstatus: " << gstatus << " x status: " << status
                  << std::endl;
      if (status < 0.)
        gstatus = -1.;
      else
        gstatus += status;
      if (verbose_)
        std::cout << "[TrackingQualityChecker::fillTrackingStatus] ===> gstatus: " << gstatus << std::endl;
      if (verbose_)
        std::cout << "[TrackingQualityChecker::fillTrackingStatus] ME: " << it->first << " ["
                  << it->second.TrackingFlag->getFullname() << "] flag: " << it->second.TrackingFlag->getFloatValue()
                  << std::endl;

    } else {  // more than 1 ME w/ the same root => they need to be considered together
      float status = 1.;
      for (auto ime : tmpMEvec) {
        float tmp_status = 1.;
        std::string name = ime->getName();
        if (name.find(MEname) != std::string::npos) {
          me = ime;

          if (verbose_)
            std::cout << "[TrackingQualityChecker::fillTrackingStatus] status: " << status << std::endl;
          std::vector<QReport*> qt_reports = me->getQReports();
          size_t nQTme = qt_reports.size();
          if (verbose_)
            std::cout << "nQTme: " << nQTme << "[" << name << ", " << ime->getFullname() << "]" << std::endl;
          if (nQTme != 0) {
            if (verbose_)
              std::cout << "[TrackingQualityChecker::fillTrackingStatus] qt_reports: " << qt_reports.size()
                        << std::endl;
            // loop on possible QTs
            for (auto iQT : qt_reports) {
              tmp_status += iQT->getQTresult();
              if (verbose_)
                std::cout << "[TrackingQualityChecker::fillTrackingStatus] iQT: " << iQT->getQRName() << std::endl;
              if (verbose_)
                std::cout << "[TrackingQualityChecker::fillTrackingStatus] MEname: " << MEname
                          << " status: " << iQT->getQTresult() << " exists ? "
                          << (it->second.TrackingFlag ? "yes " : "no ") << it->second.TrackingFlag << std::endl;
              if (verbose_)
                std::cout << "[TrackingQualityChecker::fillTrackingStatus] iQT message: " << iQT->getMessage()
                          << std::endl;
              if (verbose_)
                std::cout << "[TrackingQualityChecker::fillTrackingStatus] iQT status: " << iQT->getStatus()
                          << std::endl;
            }
            tmp_status = tmp_status / float(nQTme);
          }
        }
        status = fminf(tmp_status, status);
      }
      if (status < 0.)
        gstatus = -1.;
      else
        gstatus += status;
      if (verbose_)
        std::cout << "[TrackingQualityChecker::fillTrackingStatus] MEname: " << MEname << " status: " << status
                  << std::endl;
      it->second.TrackingFlag->Fill(status);
      if (verbose_)
        std::cout << "[TrackingQualityChecker::fillTrackingStatus] TrackGlobalSummaryReportMap: "
                  << TrackGlobalSummaryReportMap << std::endl;

      fillStatusHistogram(TrackGlobalSummaryReportMap, ibin, 1, status);
    }
  }

  // After harvesting, all per-lumi MEs are reset, to make sure we only get
  // events of the new lumisection next time.
  for (std::map<std::string, TrackingMEs>::iterator it = TrackingMEsMap.begin(); it != TrackingMEsMap.end(); it++) {
    std::string localMEdirpath = it->second.HistoDir;
    std::vector<MonitorElement*> tmpMEvec = igetter.getContents(ibooker.pwd() + "/" + localMEdirpath);
    for (auto ime : tmpMEvec) {
      if (ime->getLumiFlag()) {
        ime->Reset();
      }
    }
  }

  if (verbose_)
    std::cout << "[TrackingQualityChecker::fillTrackingStatus] gstatus: " << gstatus << std::endl;
  size_t nQT = TrackingMEsMap.size();
  if (gstatus < 1.)
    gstatus = -1.;
  else
    gstatus = gstatus / float(nQT);

  if (verbose_)
    std::cout << "[TrackingQualityChecker::fillTrackingStatus] ===> gstatus: " << gstatus << std::endl;
  TrackGlobalSummaryReportGlobal->Fill(gstatus);
  ibooker.cd();

  if (verbose_)
    std::cout << "[TrackingQualityChecker::fillTrackingStatus] DONE" << std::endl;
}

//
// -- Fill Report Summary Map
//
void TrackingQualityChecker::fillStatusHistogram(MonitorElement* me, int xbin, int ybin, float val) {
  if (me && me->kind() == MonitorElement::Kind::TH2F) {
    TH2F* th2d = me->getTH2F();
    th2d->SetBinContent(xbin, ybin, val);
  }
}

// Fill Tracking Status MEs at the Lumi block
//
void TrackingQualityChecker::fillTrackingStatusAtLumi(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  if (verbose_)
    std::cout << "[TrackingQualityChecker::fillTrackingStatusAtLumi] starting .. " << std::endl;
  float gstatus = 1.0;

  ibooker.cd();
  if (!TrackingUtility::goToDir(ibooker, igetter, TopFolderName_))
    return;

  int ibin = 0;
  for (std::map<std::string, TrackingLSMEs>::iterator it = TrackingLSMEsMap.begin(); it != TrackingLSMEsMap.end();
       it++) {
    if (verbose_)
      std::cout << "[TrackingQualityChecker::fillTrackingStatusAtLumi] ME: " << it->first << " ["
                << it->second.TrackingFlag->getFullname() << "] flag: " << it->second.TrackingFlag->getFloatValue()
                << std::endl;

    ibin++;

    std::string localMEdirpath = it->second.HistoLSDir;
    std::string MEname = it->second.HistoLSName;
    float lower_cut = it->second.HistoLSLowerCut;
    float upper_cut = it->second.HistoLSUpperCut;

    float status = 1.0;

    std::vector<MonitorElement*> tmpMEvec = igetter.getContents(ibooker.pwd() + "/" + localMEdirpath);
    if (verbose_)
      std::cout << "[TrackingQualityChecker::fillTrackingStatusAtLumi] tmpMEvec: " << tmpMEvec.size() << std::endl;

    MonitorElement* me = nullptr;

    size_t nMEs = 0;
    for (auto ime : tmpMEvec) {
      std::string name = ime->getName();
      if (name.find(MEname) != std::string::npos) {
        me = ime;
        nMEs++;
      }
    }
    // only one ME found
    if (nMEs == 1) {
      for (auto ime : tmpMEvec) {
        std::string name = ime->getName();
        if (name.find(MEname) != std::string::npos) {
          me = ime;
        }
      }
      if (!me)
        continue;

      if (me->kind() == MonitorElement::Kind::TH1F) {
        float x_mean = me->getMean();
        if (verbose_)
          std::cout << "[TrackingQualityChecker::fillTrackingStatusAtLumi] MEname: " << MEname << " x_mean: " << x_mean
                    << std::endl;
        if (x_mean <= lower_cut || x_mean > upper_cut)
          status = 0.0;
        else
          status = 1.0;
      }
    } else {  // more than 1 ME w/ the same root => they need to be considered together
      for (auto ime : tmpMEvec) {
        float tmp_status = 1.;
        std::string name = ime->getName();
        if (name.find(MEname) != std::string::npos) {
          me = ime;
          if (!me)
            continue;

          if (me->kind() == MonitorElement::Kind::TH1F) {
            float x_mean = me->getMean();
            if (verbose_)
              std::cout << "[TrackingQualityChecker::fillTrackingStatusAtLumi] MEname: " << MEname << "["
                        << me->getName() << "]  x_mean: " << x_mean << std::endl;
            if (x_mean <= lower_cut || x_mean > upper_cut)
              tmp_status = 0.0;
            else
              tmp_status = 1.0;
            if (verbose_)
              std::cout << "[TrackingQualityChecker::fillTrackingStatusAtLumi] tmp_status: " << tmp_status << std::endl;
          }
        }
        status = fminf(tmp_status, status);
        if (verbose_)
          std::cout << "[TrackingQualityChecker::fillTrackingStatusAtLumi] ==> status: " << status << std::endl;
      }  // loop on tmpMEvec
    }
    it->second.TrackingFlag->Fill(status);
    if (verbose_)
      std::cout << "[TrackingQualityChecker::fillTrackingStatusAtLumi] ===> status: " << status << " [" << gstatus
                << "]" << std::endl;
    if (status == 0.0)
      gstatus = -1.0;
    else
      gstatus = gstatus * status;
    if (verbose_)
      std::cout << "[TrackingQualityChecker::fillTrackingStatusAtLumi] ===> gstatus: " << gstatus << std::endl;
    if (verbose_)
      std::cout << "[TrackingQualityChecker::fillTrackingStatusAtLumi] ME: " << it->first << " ["
                << it->second.TrackingFlag->getFullname() << "] flag: " << it->second.TrackingFlag->getFloatValue()
                << std::endl;
  }
  TrackLSSummaryReportGlobal->Fill(gstatus);
  ibooker.cd();

  if (verbose_)
    std::cout << "[TrackingQualityChecker::fillTrackingStatusAtLumi] DONE" << std::endl;
}
