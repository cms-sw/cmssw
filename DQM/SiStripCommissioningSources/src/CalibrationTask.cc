#include "DQM/SiStripCommissioningSources/interface/CalibrationTask.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <CondFormats/DataRecord/interface/SiStripPedestalsRcd.h>
#include <CondFormats/SiStripObjects/interface/SiStripPedestals.h>

#include <arpa/inet.h>
#include <cstdio>
#include <fstream>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/unistd.h>

// -----------------------------------------------------------------------------
//
CalibrationTask::CalibrationTask(DQMStore* dqm,
                                 const FedChannelConnection& conn,
                                 const sistrip::RunType& rtype,
                                 const char* filename,
                                 uint32_t run,
                                 const edm::EventSetup& setup)
    : CommissioningTask(dqm, conn, "CalibrationTask"),
      runType_(rtype),
      nBins_(0),
      lastCalChan_(1000),
      lastCalSel_(1000),
      lastLatency_(1000),
      extrainfo_(),
      run_(run) {
  // each latency point is 25ns, each calSel is 25/8 --> 64 points = 8 calsel + 8 latency --> as set in the DAQ configuration
  if (runType_ == sistrip::CALIBRATION)
    nBins_ = 64;
  // each latency point is 25ns, each calSel is 25/8 --> 80 points = 8 calsel + 10 latency --> as set in the DAQ configuration
  else if (runType_ == sistrip::CALIBRATION_DECO)
    nBins_ = 80;

  LogDebug("Commissioning") << "[CalibrationTask::CalibrationTask] Constructing object...";

  // load the pedestals
  edm::ESHandle<SiStripPedestals> pedestalsHandle;
  setup.get<SiStripPedestalsRcd>().get(pedestalsHandle);
  SiStripPedestals::Range detPedRange = pedestalsHandle->getRange(conn.detId());
  int start = conn.apvPairNumber() * 256;
  int stop = start + 256;
  int value = 0;
  ped.reserve(256);
  for (int strip = start; strip < stop; ++strip) {
    value = int(pedestalsHandle->getPed(strip, detPedRange));
    if (value > 895)
      value -= 1024;
    ped.push_back(value);
  }
}

// -----------------------------------------------------------------------------
//
CalibrationTask::~CalibrationTask() {
  LogDebug("Commissioning") << "[CalibrationTask::CalibrationTask] Destructing object...";
}

// -----------------------------------------------------------------------------
//
void CalibrationTask::book() {
  LogDebug("Commissioning") << "[CalibrationTask::book]";

  // book 16 histograms, one for each strip in a calibration group --> APV1 --> 16 = dimension of one calChan
  if (calib1_.find(extrainfo_) == calib1_.end()) {
    dqm()->setCurrentFolder(directory_);
    calib1_[extrainfo_].resize(16);
    for (int i = 0; i < 16; ++i) {
      std::string postfix = extrainfo_ + "_istrip_" + std::to_string(i);
      std::string title = SiStripHistoTitle(sistrip::EXPERT_HISTO,
                                            runType_,
                                            sistrip::FED_KEY,
                                            fedKey(),
                                            sistrip::APV,
                                            connection().i2cAddr(0),
                                            postfix)
                              .title();

      if (runType_ == sistrip::CALIBRATION)
        calib1_[extrainfo_][i].histo(dqm()->book1D(title, title, nBins_, 0, 200));
      else if (runType_ == sistrip::CALIBRATION_DECO)
        calib1_[extrainfo_][i].histo(dqm()->book1D(title, title, nBins_, 0, 250));
      calib1_[extrainfo_][i].isProfile_ = false;
      calib1_[extrainfo_][i].vNumOfEntries_.resize(nBins_, 0);
    }
  }

  // book 16 histograms, one for each strip in a calibration group --> APV2  --> 16 = dimension of one calChan
  if (calib2_.find(extrainfo_) == calib2_.end()) {
    dqm()->setCurrentFolder(directory_);
    calib2_[extrainfo_].resize(16);
    for (int i = 0; i < 16; ++i) {
      std::string postfix = extrainfo_ + "_istrip_" + std::to_string(i);
      std::string title = SiStripHistoTitle(sistrip::EXPERT_HISTO,
                                            runType_,
                                            sistrip::FED_KEY,
                                            fedKey(),
                                            sistrip::APV,
                                            connection().i2cAddr(1),
                                            postfix)
                              .title();

      if (runType_ == sistrip::CALIBRATION)
        calib2_[extrainfo_][i].histo(dqm()->book1D(title, title, nBins_, 0, 200));
      else if (runType_ == sistrip::CALIBRATION_DECO)
        calib2_[extrainfo_][i].histo(dqm()->book1D(title, title, nBins_, 0, 250));

      calib2_[extrainfo_][i].isProfile_ = false;
      calib2_[extrainfo_][i].vNumOfEntries_.resize(nBins_, 0);
    }
  }
}

// -----------------------------------------------------------------------------
//
void CalibrationTask::fill(const SiStripEventSummary& summary, const edm::DetSet<SiStripRawDigi>& digis) {
  LogDebug("Commissioning") << "[CalibrationTask::fill]";

  if (lastCalChan_ != summary.calChan()) {  // change in the calChan value
    lastCalChan_ = summary.calChan();
    lastLatency_ = summary.latency();
    lastCalSel_ = summary.calSel();
    extrainfo_ = "calChan_" + std::to_string(lastCalChan_);
    book();  // book histograms and load the right one
  } else {
    lastCalChan_ = summary.calChan();
    lastLatency_ = summary.latency();
    lastCalSel_ = summary.calSel();
    extrainfo_ = "calChan_" + std::to_string(lastCalChan_);
  }

  // Check if CalChan changed. In that case, save, reset histo, change title, and continue
  int isub = lastCalChan_ < 4 ? lastCalChan_ + 4 : lastCalChan_ - 4;
  int bin = 0;

  if (runType_ == sistrip::CALIBRATION)
    bin = (100 - summary.latency()) * 8 + (7 - summary.calSel());
  else if (runType_ == sistrip::CALIBRATION_DECO)
    bin = (102 - summary.latency()) * 8 + (7 - summary.calSel());

  // Fill the histograms data-ped -(data-ped)_isub, the second term corresponds to the common mode substraction, looking at a strip far away.
  for (int k = 0; k < 16; ++k) {
    updateHistoSet(calib1_[extrainfo_][k],
                   bin,
                   digis.data[lastCalChan_ + k * 8].adc() - ped[lastCalChan_ + k * 8] -
                       (digis.data[isub + k * 8].adc() - ped[isub + k * 8]));
    updateHistoSet(calib2_[extrainfo_][k],
                   bin,
                   digis.data[128 + lastCalChan_ + k * 8].adc() - ped[128 + lastCalChan_ + k * 8] -
                       (digis.data[128 + isub + k * 8].adc() - ped[128 + isub + k * 8]));
  }
  update();  //TODO: temporary: find a better solution later
}

// -----------------------------------------------------------------------------
//
void CalibrationTask::update() {
  LogDebug("Commissioning") << "[CalibrationTask::update]";  // huge output

  for (const auto& element : calib1_) {            // all pulse for different calChan
    for (auto vecelement : element.second)  // all strips in a calCan
      updateHistoSet(vecelement);
  }

  for (const auto& element : calib2_) {            // all pulse for different calChan
    for (auto vecelement : element.second)  // all strips in a calCan
      updateHistoSet(vecelement);
  }
}

void CalibrationTask::setCurrentFolder(const std::string& dir) { directory_ = dir; }
