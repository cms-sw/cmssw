#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/TrackingMonitorClient/interface/TrackingUtility.h"
#include "DQM/TrackingMonitorClient/interface/TrackingCertificationInfo.h"

#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

//Run Info
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"

#include <iomanip>
#include <cstdio>
#include <string>
#include <sstream>
#include <cmath>
//
// -- Contructor
//
TrackingCertificationInfo::TrackingCertificationInfo(edm::ParameterSet const& pSet)
    : pSet_(pSet),
      trackingCertificationBooked_(false),
      trackingLSCertificationBooked_(false),
      nFEDConnected_(0),
      allPixelFEDConnected_(true),
      m_cacheID_(0),
      runInfoToken_(esConsumes<RunInfo, RunInfoRcd, edm::Transition::BeginRun>()),
      detCablingToken_(esConsumes<SiStripDetCabling, SiStripDetCablingRcd, edm::Transition::BeginRun>()) {
  // Create MessageSender
  edm::LogInfo("TrackingCertificationInfo") << "TrackingCertificationInfo::Deleting TrackingCertificationInfo ";

  verbose_ = pSet_.getUntrackedParameter<bool>("verbose", false);
  TopFolderName_ = pSet_.getUntrackedParameter<std::string>("TopFolderName", "Tracking");
  if (verbose_)
    edm::LogInfo("TrackingCertificationInfo") << "TopFolderName_: " << TopFolderName_ << std::endl;

  TrackingMEs tracking_mes;
  // load variables for Global certification

  checkPixelFEDs_ = pSet_.getParameter<bool>("checkPixelFEDs");
  if (checkPixelFEDs_) {
    std::string QTname = "pixel";
    tracking_mes.TrackingFlag = nullptr;
    TrackingMEsMap.insert(std::pair<std::string, TrackingMEs>(QTname, tracking_mes));
  }

  std::vector<edm::ParameterSet> TrackingGlobalQualityMEs =
      pSet_.getParameter<std::vector<edm::ParameterSet> >("TrackingGlobalQualityPSets");
  for (const auto& meQTset : TrackingGlobalQualityMEs) {
    std::string QTname = meQTset.getParameter<std::string>("QT");
    tracking_mes.TrackingFlag = nullptr;

    if (verbose_)
      edm::LogInfo("TrackingCertificationInfo") << " inserting " << QTname << " in TrackingMEsMap" << std::endl;
    TrackingMEsMap.insert(std::pair<std::string, TrackingMEs>(QTname, tracking_mes));
  }

  TrackingLSMEs tracking_ls_mes;
  // load variables for LS certification
  std::vector<edm::ParameterSet> TrackingLSQualityMEs =
      pSet_.getParameter<std::vector<edm::ParameterSet> >("TrackingLSQualityMEs");
  for (const auto& meQTset : TrackingLSQualityMEs) {
    std::string QTname = meQTset.getParameter<std::string>("QT");
    tracking_ls_mes.TrackingFlag = nullptr;

    if (verbose_)
      edm::LogInfo("TrackingCertificationInfo") << " inserting " << QTname << " in TrackingMEsMap" << std::endl;
    TrackingLSMEsMap.insert(std::pair<std::string, TrackingLSMEs>(QTname, tracking_ls_mes));
  }

  // define sub-detectors which affect the quality
  SubDetFolder.push_back("SiStrip");
  SubDetFolder.push_back("Pixel");
}

TrackingCertificationInfo::~TrackingCertificationInfo() {
  edm::LogInfo("TrackingCertificationInfo") << "Deleting TrackingCertificationInfo ";
}
//
// -- Begin Job
//
void TrackingCertificationInfo::beginJob() {}
//
// -- Begin Run
//
void TrackingCertificationInfo::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) {
  edm::LogInfo("TrackingCertificationInfo") << "beginRun starting .." << std::endl;

  detCabling_ = &(eSetup.getData(detCablingToken_));

  unsigned long long cacheID = eSetup.get<SiStripDetCablingRcd>().cacheIdentifier();
  if (m_cacheID_ != cacheID) {
    m_cacheID_ = cacheID;
  }

  nFEDConnected_ = 0;
  int nPixelFEDConnected_ = 0;
  const int siStripFedIdMin = FEDNumbering::MINSiStripFEDID;
  const int siStripFedIdMax = FEDNumbering::MAXSiStripFEDID;
  const int siPixelFedIdMin = FEDNumbering::MINSiPixelFEDID;
  const int siPixelFedIdMax = FEDNumbering::MAXSiPixelFEDID;
  const int siPixelFeds = (siPixelFedIdMax - siPixelFedIdMin + 1);

  edm::ESHandle<RunInfo> runInfoRec = eSetup.getHandle(runInfoToken_);
  if (runInfoRec.isValid()) {
    sumFED_ = runInfoRec.product();
    if (sumFED_ != nullptr) {
      std::vector<int> FedsInIds = sumFED_->m_fed_in;
      for (auto fedID : FedsInIds) {
        if (fedID >= siPixelFedIdMin && fedID <= siPixelFedIdMax) {
          ++nFEDConnected_;
          ++nPixelFEDConnected_;
        } else if (fedID >= siStripFedIdMin && fedID <= siStripFedIdMax)
          ++nFEDConnected_;
      }
      LogDebug("TrackingDcsInfo") << " TrackingDcsInfo :: Connected FEDs " << nFEDConnected_;
    }
  }

  allPixelFEDConnected_ = (nPixelFEDConnected_ == siPixelFeds);
}

//
// -- Book MEs for Tracking Certification fractions
//
void TrackingCertificationInfo::bookTrackingCertificationMEs(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_) {
  if (verbose_)
    edm::LogInfo("TrackingCertificationInfo")
        << "bookTrackingCertificationMEs starting .. trackingCertificationBooked_: " << trackingCertificationBooked_
        << std::endl;

  if (!trackingCertificationBooked_) {
    ibooker_.cd();
    std::string tracking_dir = "";
    TrackingUtility::getTopFolderPath(ibooker_, igetter_, TopFolderName_, tracking_dir);

    if (!tracking_dir.empty())
      ibooker_.setCurrentFolder(tracking_dir + "/EventInfo");
    else
      ibooker_.setCurrentFolder(TopFolderName_ + "/EventInfo");

    TrackingCertification = ibooker_.bookFloat("CertificationSummary");

    std::string hname, htitle;
    hname = "CertificationReportMap";
    htitle = "Tracking Certification Summary Map";
    size_t nQT = TrackingMEsMap.size();
    TrackingCertificationSummaryMap = ibooker_.book2D(hname, htitle, nQT, 0.5, float(nQT) + 0.5, 1, 0.5, 1.5);
    TrackingCertificationSummaryMap->setAxisTitle("Track Quality Type", 1);
    TrackingCertificationSummaryMap->setAxisTitle("QTest Flag", 2);
    size_t ibin = 0;
    for (const auto& meQTset : TrackingMEsMap) {
      TrackingCertificationSummaryMap->setBinLabel(ibin + 1, meQTset.first);
      ibin++;
    }

    if (!tracking_dir.empty())
      ibooker_.setCurrentFolder(TopFolderName_ + "/EventInfo/CertificationContents");
    else
      ibooker_.setCurrentFolder(TopFolderName_ + "/EventInfo/CertificationContents");

    for (std::map<std::string, TrackingMEs>::iterator it = TrackingMEsMap.begin(); it != TrackingMEsMap.end(); it++) {
      std::string meQTname = it->first;
      if (verbose_)
        edm::LogInfo("TrackingCertificationInfo") << "bookStatus meQTname: " << meQTname << std::endl;
      it->second.TrackingFlag = ibooker_.bookFloat("Track" + meQTname);
      if (verbose_)
        edm::LogInfo("TrackingCertificationInfo")
            << "bookStatus " << it->first << " exists ? " << it->second.TrackingFlag << std::endl;
    }

    trackingCertificationBooked_ = true;
    ibooker_.cd();
  }

  if (verbose_)
    edm::LogInfo("TrackingCertificationInfo")
        << "bookStatus trackingCertificationBooked_: " << trackingCertificationBooked_ << std::endl;
}

//
// -- Book MEs for Tracking Certification per LS
//
void TrackingCertificationInfo::bookTrackingCertificationMEsAtLumi(DQMStore::IBooker& ibooker_,
                                                                   DQMStore::IGetter& igetter_) {
  if (verbose_)
    edm::LogInfo("TrackingCertificationInfo")
        << "bookTrackingCertificationMEs starting .. trackingCertificationBooked_: " << trackingCertificationBooked_
        << std::endl;

  if (!trackingLSCertificationBooked_) {
    ibooker_.cd();
    std::string tracking_dir = "";
    TrackingUtility::getTopFolderPath(ibooker_, igetter_, TopFolderName_, tracking_dir);

    if (!tracking_dir.empty())
      ibooker_.setCurrentFolder(tracking_dir + "/EventInfo");
    else
      ibooker_.setCurrentFolder(TopFolderName_ + "/EventInfo");

    TrackingLSCertification = ibooker_.bookFloat("CertificationSummary");

    if (!tracking_dir.empty())
      ibooker_.setCurrentFolder(TopFolderName_ + "/EventInfo/CertificationContents");
    else
      ibooker_.setCurrentFolder(TopFolderName_ + "/EventInfo/CertificationContents");

    for (std::map<std::string, TrackingLSMEs>::iterator it = TrackingLSMEsMap.begin(); it != TrackingLSMEsMap.end();
         it++) {
      std::string meQTname = it->first;
      if (verbose_)
        edm::LogInfo("TrackingCertificationInfo") << "bookStatus meQTname: " << meQTname << std::endl;
      it->second.TrackingFlag = ibooker_.bookFloat("Track" + meQTname);
      if (verbose_)
        edm::LogInfo("TrackingCertificationInfo")
            << "bookStatus " << it->first << " exists ? " << it->second.TrackingFlag << std::endl;
    }

    trackingLSCertificationBooked_ = true;
    ibooker_.cd();
  }

  if (verbose_)
    edm::LogInfo("TrackingCertificationInfo")
        << "[TrackingCertificationInfo::bookStatus] trackingCertificationBooked_: " << trackingCertificationBooked_
        << std::endl;
}
//

//
// -- End Luminosity Block
//
void TrackingCertificationInfo::dqmEndLuminosityBlock(DQMStore::IBooker& ibooker_,
                                                      DQMStore::IGetter& igetter_,
                                                      edm::LuminosityBlock const& lumiSeg,
                                                      edm::EventSetup const& eSetup) {
  edm::LogInfo("TrackingCertificationInfo") << "endLuminosityBlock";
  bookTrackingCertificationMEsAtLumi(ibooker_, igetter_);
  fillDummyTrackingCertificationAtLumi(ibooker_, igetter_);

  if (nFEDConnected_ > 0)
    fillTrackingCertificationMEsAtLumi(ibooker_, igetter_);
  else
    fillDummyTrackingCertificationAtLumi(ibooker_, igetter_);
}

//
// -- End of Run
//
void TrackingCertificationInfo::dqmEndJob(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_) {
  edm::LogInfo("TrackingCertificationInfo") << "dqmEndJob" << std::endl;

  bookTrackingCertificationMEs(ibooker_, igetter_);
  fillDummyTrackingCertification(ibooker_, igetter_);

  if (nFEDConnected_ > 0)
    fillTrackingCertificationMEs(ibooker_, igetter_);
  else
    fillDummyTrackingCertification(ibooker_, igetter_);

  edm::LogInfo("TrackingCertificationInfo") << "[TrackingCertificationInfo::endRun] DONE" << std::endl;
}
//
// --Fill Tracking Certification
//
void TrackingCertificationInfo::fillTrackingCertificationMEs(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_) {
  if (!trackingCertificationBooked_) {
    edm::LogError("TrackingCertificationInfo") << "fillTrackingCertificationMEs : MEs missing " << std::endl;
    return;
  }

  ibooker_.cd();
  std::string tracking_dir = "";
  TrackingUtility::getTopFolderPath(ibooker_, igetter_, TopFolderName_, tracking_dir);
  if (verbose_)
    edm::LogInfo("TrackingCertificationInfo")
        << "fillTrackingCertificationMEs tracking_dir: " << tracking_dir << std::endl;
  std::vector<MonitorElement*> all_mes = igetter_.getContents(tracking_dir + "/EventInfo/reportSummaryContents");
  float fval = 1.0;

  if (verbose_)
    edm::LogInfo("TrackingCertificationInfo") << "all_mes: " << all_mes.size() << std::endl;

  if (checkPixelFEDs_) {
    float val = 1.;
    if (allPixelFEDConnected_)
      val = 0.;
    int xbin = 0;
    for (std::map<std::string, TrackingMEs>::const_iterator it = TrackingMEsMap.begin(); it != TrackingMEsMap.end();
         it++) {
      std::string type = it->first;
      if (type == "pixel") {
        it->second.TrackingFlag->Fill(val);
        TH2F* th2d = TrackingCertificationSummaryMap->getTH2F();
        if (verbose_)
          edm::LogInfo("TrackingCertificationInfo")
              << "fillTrackingCertificationMEs xbin: " << xbin << " val: " << val << std::endl;
        th2d->SetBinContent(xbin + 1, 1, val);
      }
      xbin++;
    }
    fval = fminf(fval, val);
  }

  int xbin = (checkPixelFEDs_ ? 1 : 0);
  for (std::vector<MonitorElement*>::const_iterator ime = all_mes.begin(); ime != all_mes.end(); ime++) {
    MonitorElement* me = (*ime);
    if (!me)
      continue;
    if (verbose_)
      edm::LogInfo("TrackingCertificationInfo") << "fillTrackingCertificationMEs me: " << me->getName() << std::endl;
    if (me->kind() == MonitorElement::Kind::REAL) {
      const std::string& name = me->getName();
      float val = me->getFloatValue();

      for (std::map<std::string, TrackingMEs>::const_iterator it = TrackingMEsMap.begin(); it != TrackingMEsMap.end();
           it++) {
        if (verbose_)
          edm::LogInfo("TrackingCertificationInfo")
              << "fillTrackingCertificationMEs ME: " << it->first << " [" << it->second.TrackingFlag->getFullname()
              << "] flag: " << it->second.TrackingFlag->getFloatValue() << std::endl;

        std::string type = it->first;
        if (name.find(type) != std::string::npos) {
          if (verbose_)
            edm::LogInfo("TrackingCertificationInfo")
                << "fillTrackingCertificationMEs type: " << type << " <---> name: " << name << std::endl;
          it->second.TrackingFlag->Fill(val);
          if (verbose_)
            edm::LogInfo("TrackingCertificationInfo")
                << "fillTrackingCertificationMEs xbin: " << xbin << " val: " << val << std::endl;
          TH2F* th2d = TrackingCertificationSummaryMap->getTH2F();
          th2d->SetBinContent(xbin + 1, 1, val);
          xbin++;
          break;
        }
        if (verbose_)
          edm::LogInfo("TrackingCertificationInfo")
              << "[TrackingCertificationInfo::fillTrackingCertificationMEs] ME: " << it->first << " ["
              << it->second.TrackingFlag->getFullname() << "] flag: " << it->second.TrackingFlag->getFloatValue()
              << std::endl;
      }
      fval = fminf(fval, val);
    }
  }
  if (verbose_)
    edm::LogInfo("TrackingCertificationInfo")
        << "fillTrackingCertificationMEs TrackingCertification: " << fval << std::endl;
  TrackingCertification->Fill(fval);
}

//
// --Reset Tracking Certification
//
void TrackingCertificationInfo::resetTrackingCertificationMEs(DQMStore::IBooker& ibooker_,
                                                              DQMStore::IGetter& igetter_) {
  if (!trackingCertificationBooked_)
    bookTrackingCertificationMEs(ibooker_, igetter_);
  TrackingCertification->Reset();
  for (std::map<std::string, TrackingMEs>::const_iterator it = TrackingMEsMap.begin(); it != TrackingMEsMap.end();
       it++) {
    it->second.TrackingFlag->Reset();
  }
}

//
// --Reset Tracking Certification per LS
//
void TrackingCertificationInfo::resetTrackingCertificationMEsAtLumi(DQMStore::IBooker& ibooker_,
                                                                    DQMStore::IGetter& igetter_) {
  if (!trackingLSCertificationBooked_)
    bookTrackingCertificationMEsAtLumi(ibooker_, igetter_);
  TrackingLSCertification->Reset();
  for (std::map<std::string, TrackingLSMEs>::const_iterator it = TrackingLSMEsMap.begin(); it != TrackingLSMEsMap.end();
       it++) {
    it->second.TrackingFlag->Reset();
  }
}

//
// -- Fill Dummy Tracking Certification
//
void TrackingCertificationInfo::fillDummyTrackingCertification(DQMStore::IBooker& ibooker_,
                                                               DQMStore::IGetter& igetter_) {
  resetTrackingCertificationMEs(ibooker_, igetter_);
  if (trackingCertificationBooked_) {
    TrackingCertification->Fill(-1.0);
    for (std::map<std::string, TrackingMEs>::const_iterator it = TrackingMEsMap.begin(); it != TrackingMEsMap.end();
         it++) {
      it->second.TrackingFlag->Fill(-1.0);
    }

    for (int xbin = 1; xbin < TrackingCertificationSummaryMap->getNbinsX() + 1; xbin++)
      for (int ybin = 1; ybin < TrackingCertificationSummaryMap->getNbinsY() + 1; ybin++)
        TrackingCertificationSummaryMap->Fill(xbin, ybin, -1);
  }
}

//
// -- Fill Dummy Tracking Certification per LS
//
void TrackingCertificationInfo::fillDummyTrackingCertificationAtLumi(DQMStore::IBooker& ibooker_,
                                                                     DQMStore::IGetter& igetter_) {
  resetTrackingCertificationMEsAtLumi(ibooker_, igetter_);
  if (trackingLSCertificationBooked_) {
    TrackingLSCertification->Fill(-1.0);
    for (std::map<std::string, TrackingLSMEs>::const_iterator it = TrackingLSMEsMap.begin();
         it != TrackingLSMEsMap.end();
         it++) {
      it->second.TrackingFlag->Fill(-1.0);
    }
  }
}

//
// --Fill Tracking Certification per LS
//
void TrackingCertificationInfo::fillTrackingCertificationMEsAtLumi(DQMStore::IBooker& ibooker_,
                                                                   DQMStore::IGetter& igetter_) {
  if (verbose_)
    edm::LogInfo("TrackingCertificationInfo") << "fillTrackingCertificationMEsAtLumi starting .." << std::endl;
  if (!trackingLSCertificationBooked_) {
    return;
  }
  resetTrackingCertificationMEsAtLumi(ibooker_, igetter_);

  ibooker_.cd();
  std::string tracking_dir = "";
  TrackingUtility::getTopFolderPath(ibooker_, igetter_, TopFolderName_, tracking_dir);
  if (verbose_)
    edm::LogInfo("TrackingCertificationInfo")
        << "fillTrackingCertificationMEsAtLumi tracking_dir: " << tracking_dir << std::endl;

  if (verbose_)
    edm::LogInfo("TrackingCertificationInfo")
        << "fillTrackingCertificationMEsAtLumi tracking_dir: " << tracking_dir << std::endl;
  std::vector<MonitorElement*> all_mes = igetter_.getContents(tracking_dir + "/EventInfo/reportSummaryContents");

  if (verbose_)
    edm::LogInfo("TrackingCertificationInfo") << "all_mes: " << all_mes.size() << std::endl;

  for (std::vector<MonitorElement*>::const_iterator ime = all_mes.begin(); ime != all_mes.end(); ime++) {
    MonitorElement* me = (*ime);
    if (!me)
      continue;
    if (verbose_)
      edm::LogInfo("TrackingCertificationInfo")
          << "fillTrackingCertificationMEsAtLumi me: " << me->getName() << std::endl;
    if (me->kind() == MonitorElement::Kind::REAL) {
      const std::string& name = me->getName();
      float val = me->getFloatValue();
      if (verbose_)
        edm::LogInfo("TrackingCertificationInfo") << "fillTrackingCertificationMEsAtLumi val:  " << val << std::endl;

      for (std::map<std::string, TrackingLSMEs>::const_iterator it = TrackingLSMEsMap.begin();
           it != TrackingLSMEsMap.end();
           it++) {
        if (verbose_)
          edm::LogInfo("TrackingCertificationInfo")
              << "fillTrackingCertificationMEsAtLumi ME: " << it->first << " ["
              << it->second.TrackingFlag->getFullname() << "] flag: " << it->second.TrackingFlag->getFloatValue()
              << std::endl;

        std::string type = it->first;
        if (verbose_)
          edm::LogInfo("TrackingCertificationInfo") << "fillTrackingCertificationMEsAtLumi type: " << type << std::endl;
        if (name.find(type) != std::string::npos) {
          if (verbose_)
            edm::LogInfo("TrackingCertificationInfo")
                << "fillTrackingCertificationMEsAtLumi type: " << type << " <---> name: " << name << std::endl;
          it->second.TrackingFlag->Fill(val);
          break;
        }
        if (verbose_)
          edm::LogInfo("TrackingCertificationInfo")
              << "fillTrackingCertificationMEsAtLumi ME: " << it->first << " ["
              << it->second.TrackingFlag->getFullname() << "] flag: " << it->second.TrackingFlag->getFloatValue()
              << std::endl;
      }
    }
  }

  float global_dqm_flag = 1.0;
  std::string full_path = tracking_dir + "/EventInfo/reportSummary";
  MonitorElement* me_dqm = igetter_.get(full_path);
  if (me_dqm && me_dqm->kind() == MonitorElement::Kind::REAL)
    global_dqm_flag = me_dqm->getFloatValue();
  if (verbose_)
    edm::LogInfo("TrackingCertificationInfo")
        << "fillTrackingCertificationMEsAtLumi global_dqm_flag: " << global_dqm_flag << std::endl;

  TrackingLSCertification->Reset();
  TrackingLSCertification->Fill(global_dqm_flag);
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackingCertificationInfo);
