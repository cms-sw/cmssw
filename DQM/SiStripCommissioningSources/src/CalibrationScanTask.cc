#include "DQM/SiStripCommissioningSources/interface/CalibrationScanTask.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <CondFormats/SiStripObjects/interface/SiStripPedestals.h>

#include <arpa/inet.h>
#include <cstdio>
#include <fstream>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/unistd.h>

// -----------------------------------------------------------------------------
//
CalibrationScanTask::CalibrationScanTask(DQMStore* dqm,
                                         const FedChannelConnection& conn,
                                         const sistrip::RunType& rtype,
                                         const char* filename,
                                         uint32_t run,
                                         const SiStripPedestals& pedestals)
    : CommissioningTask(dqm, conn, "CalibrationScanTask"),
      runType_(rtype),
      nBins_(0),
      lastISHA_(1000),
      lastVFS_(1000),
      lastCalChan_(1000),
      lastCalSel_(1000),
      lastLatency_(1000),
      extrainfo_(),
      run_(run) {
  // each latency point is 25ns, each calSel is 25/8 --> 64 points = 8 calsel + 8 latency --> as set in the DAQ configuration
  if (runType_ == sistrip::CALIBRATION_SCAN)
    nBins_ = 64;
  // each latency point is 25ns, each calSel is 25/8 --> 80 points = 8 calsel + 10 latency --> as set in the DAQ configuration
  else if (runType_ == sistrip::CALIBRATION_SCAN_DECO)
    nBins_ = 80;

  LogDebug("Commissioning") << "[CalibrationScanTask::CalibrationScanTask] Constructing object...";
  // load the pedestals
  SiStripPedestals::Range detPedRange = pedestals.getRange(conn.detId());
  int start = conn.apvPairNumber() * 256;
  int stop = start + 256;
  int value = 0;
  ped.reserve(256);
  LogDebug("Commissioning") << "[CalibrationScanTask::CalibrationScanTask] Loading pedestal for " << conn.detId();
  if (conn.detId() == 0)
    return;
  for (int strip = start; strip < stop; ++strip) {
    value = int(pedestals.getPed(strip, detPedRange));
    if (value > 895)
      value -= 1024;
    ped.push_back(value);
  }
}

// -----------------------------------------------------------------------------
//
CalibrationScanTask::~CalibrationScanTask() {
  LogDebug("Commissioning") << "[CalibrationScanTask::CalibrationScanTask] Destructing object...";
}

// -----------------------------------------------------------------------------
//
void CalibrationScanTask::book() {
  if (calib1_.find(extrainfo_) == calib1_.end()) {
    // construct the histo titles and book two histograms: one per APV --> all strip in one APV are considered to fill histogram
    std::string title = SiStripHistoTitle(sistrip::EXPERT_HISTO,
                                          runType_,
                                          sistrip::FED_KEY,
                                          fedKey(),
                                          sistrip::APV,
                                          connection().i2cAddr(0),
                                          extrainfo_)
                            .title();

    dqm()->setCurrentFolder(directory_);
    calib1_[extrainfo_] = HistoSet();
    if (runType_ == sistrip::CALIBRATION_SCAN)
      calib1_[extrainfo_].histo(dqm()->book1D(title, title, nBins_, 0, 200));
    else if (runType_ == sistrip::CALIBRATION_SCAN_DECO)
      calib1_[extrainfo_].histo(dqm()->book1D(title, title, nBins_, 0, 250));
    calib1_[extrainfo_].isProfile_ = false;
    calib1_[extrainfo_].vNumOfEntries_.resize(nBins_, 0);
  }

  if (calib2_.find(extrainfo_) == calib2_.end()) {
    std::string title = SiStripHistoTitle(sistrip::EXPERT_HISTO,
                                          runType_,
                                          sistrip::FED_KEY,
                                          fedKey(),
                                          sistrip::APV,
                                          connection().i2cAddr(1),
                                          extrainfo_)
                            .title();

    dqm()->setCurrentFolder(directory_);
    calib2_[extrainfo_] = HistoSet();
    if (runType_ == sistrip::CALIBRATION_SCAN)
      calib2_[extrainfo_].histo(dqm()->book1D(title, title, nBins_, 0, 200));
    else if (runType_ == sistrip::CALIBRATION_SCAN_DECO)
      calib2_[extrainfo_].histo(dqm()->book1D(title, title, nBins_, 0, 250));
    calib2_[extrainfo_].isProfile_ = false;
    calib2_[extrainfo_].vNumOfEntries_.resize(nBins_, 0);
  }
}

// -----------------------------------------------------------------------------
//
void CalibrationScanTask::fill(const SiStripEventSummary& summary, const edm::DetSet<SiStripRawDigi>& digis) {
  if (lastISHA_ != summary.isha() or
      lastVFS_ != summary.vfs()) {  // triggered only when there is a change in isha and vfs
    lastISHA_ = summary.isha();
    lastVFS_ = summary.vfs();
    lastCalSel_ = summary.calSel();
    lastLatency_ = summary.latency();
    lastCalChan_ = summary.calChan();
    extrainfo_ = "isha_" + std::to_string(lastISHA_) + "_vfs_" + std::to_string(lastVFS_);
    book();  // book histograms and load the right one
  } else {
    lastISHA_ = summary.isha();
    lastVFS_ = summary.vfs();
    lastCalSel_ = summary.calSel();
    lastLatency_ = summary.latency();
    lastCalChan_ = summary.calChan();
    extrainfo_ = "isha_" + std::to_string(lastISHA_) + "_vfs_" + std::to_string(lastVFS_);
  }

  // retrieve the delay from the EventSummary
  int bin = 0;
  if (runType_ == sistrip::CALIBRATION_SCAN)
    bin = (100 - summary.latency()) * 8 + (7 - summary.calSel());
  else if (runType_ == sistrip::CALIBRATION_SCAN_DECO)
    bin = (102 - summary.latency()) * 8 + (7 - summary.calSel());

  // Digis are obtained for an APV pair.strips 0->127  : calib1_ strips 128->255: calib2_
  // then, only some strips are fired at a time, we use calChan to know that
  int isub = lastCalChan_ < 4 ? lastCalChan_ + 4 : lastCalChan_ - 4;
  for (int k = 0; k < 16; ++k) {
    // all strips of the APV are merged in
    updateHistoSet(calib1_[extrainfo_],
                   bin,
                   digis.data[lastCalChan_ + k * 8].adc() - ped[lastCalChan_ + k * 8] -
                       (digis.data[isub + k * 8].adc() - ped[isub + k * 8]));
    updateHistoSet(calib2_[extrainfo_],
                   bin,
                   digis.data[128 + lastCalChan_ + k * 8].adc() - ped[128 + lastCalChan_ + k * 8] -
                       (digis.data[128 + isub + k * 8].adc() - ped[128 + isub + k * 8]));
  }
  update();  //TODO: temporary: find a better solution later
}

// -----------------------------------------------------------------------------
//
void CalibrationScanTask::update() {
  LogDebug("Commissioning") << "[CalibrationScanTask::update]";  // huge output
  for (auto element : calib1_)
    updateHistoSet(element.second);
  for (auto element : calib2_)
    updateHistoSet(element.second);
}

void CalibrationScanTask::setCurrentFolder(const std::string& dir) { directory_ = dir; }
