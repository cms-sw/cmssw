#include "DQM/SiStripCommissioningSources/interface/DaqScopeModeTask.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "DQM/SiStripCommon/interface/UpdateTProfile.h"

using namespace sistrip;

// -----------------------------------------------------------------------------
//
DaqScopeModeTask::DaqScopeModeTask(DQMStore* dqm, const FedChannelConnection& conn, const edm::ParameterSet& pset)
    : CommissioningTask(dqm, conn, "DaqScopeModeTask"),
      scopeFrame_(),
      nBins_(256),     //@@ number of strips per FED channel
      nBinsSpy_(298),  //@@ in case of spy events includes header and trailing tick
      parameters_(pset) {}

// -----------------------------------------------------------------------------
//
DaqScopeModeTask::~DaqScopeModeTask() {}

// -----------------------------------------------------------------------------
//
void DaqScopeModeTask::book() {
  LogTrace(mlDqmSource_) << "[CommissioningTask::" << __func__ << "]";

  std::string title;
  if (not(parameters_.existsAs<bool>("isSpy") and parameters_.getParameter<bool>("isSpy"))) {
    //// Scope mode histograms
    title = SiStripHistoTitle(sistrip::EXPERT_HISTO,
                              sistrip::DAQ_SCOPE_MODE,
                              sistrip::FED_KEY,
                              fedKey(),
                              sistrip::LLD_CHAN,
                              connection().lldChannel(),
                              sistrip::extrainfo::scopeModeFrame_)
                .title();

    scopeFrame_.histo(dqm()->book1D(title, title, nBins_, -0.5, nBins_ - 0.5));

    scopeFrame_.vNumOfEntries_.resize(nBins_, 0);
    scopeFrame_.vSumOfContents_.resize(nBins_, 0);
    scopeFrame_.vSumOfSquares_.resize(nBins_, 0);
    scopeFrame_.isProfile_ = false;
  } else {  // use spy run to measure pedestal and tick height

    std::string extra_info;
    peds_.resize(2);
    extra_info = sistrip::extrainfo::pedestals_;

    peds_[0].isProfile_ = true;
    title = SiStripHistoTitle(sistrip::EXPERT_HISTO,
                              sistrip::DAQ_SCOPE_MODE,
                              sistrip::FED_KEY,
                              fedKey(),
                              sistrip::LLD_CHAN,
                              connection().lldChannel(),
                              extra_info)
                .title();

    peds_[0].histo(dqm()->bookProfile(title, title, nBins_, -0.5, nBins_ * 1. - 0.5, 1025, 0., 1025.));

    peds_[0].vNumOfEntries_.resize(nBins_, 0);
    peds_[0].vSumOfContents_.resize(nBins_, 0);
    peds_[0].vSumOfSquares_.resize(nBins_, 0);

    // Noise histogram
    extra_info = sistrip::extrainfo::noise_;
    peds_[1].isProfile_ = true;

    title = SiStripHistoTitle(sistrip::EXPERT_HISTO,
                              sistrip::DAQ_SCOPE_MODE,
                              sistrip::FED_KEY,
                              fedKey(),
                              sistrip::LLD_CHAN,
                              connection().lldChannel(),
                              extra_info)
                .title();

    peds_[1].histo(dqm()->bookProfile(title, title, nBins_, -0.5, nBins_ * 1. - 0.5, 1025, 0., 1025.));

    peds_[1].vNumOfEntries_.resize(nBins_, 0);
    peds_[1].vSumOfContents_.resize(nBins_, 0);
    peds_[1].vSumOfSquares_.resize(nBins_, 0);

    // Common mode histograms
    cm_.resize(2);
    int nbins = 1024;

    for (uint16_t iapv = 0; iapv < 2; iapv++) {
      title = SiStripHistoTitle(sistrip::EXPERT_HISTO,
                                sistrip::DAQ_SCOPE_MODE,
                                sistrip::FED_KEY,
                                fedKey(),
                                sistrip::APV,
                                connection().i2cAddr(iapv),
                                sistrip::extrainfo::commonMode_)
                  .title();

      cm_[iapv].histo(dqm()->book1D(title, title, nbins, -0.5, nbins * 1. - 0.5));
      cm_[iapv].isProfile_ = false;

      cm_[iapv].vNumOfEntries_.resize(nbins, 0);
      cm_[iapv].vNumOfEntries_.resize(nbins, 0);
    }

    // high and low header histograms
    title = SiStripHistoTitle(sistrip::EXPERT_HISTO,
                              sistrip::DAQ_SCOPE_MODE,
                              sistrip::FED_KEY,
                              fedKey(),
                              sistrip::LLD_CHAN,
                              connection().lldChannel(),
                              sistrip::extrainfo::scopeModeHeaderLow_)
                .title();

    lowHeader_.histo(dqm()->book1D(title, title, nbins, -0.5, 1024 * 1. - 0.5));
    lowHeader_.isProfile_ = false;
    lowHeader_.vNumOfEntries_.resize(nbins, 0);

    title = SiStripHistoTitle(sistrip::EXPERT_HISTO,
                              sistrip::DAQ_SCOPE_MODE,
                              sistrip::FED_KEY,
                              fedKey(),
                              sistrip::LLD_CHAN,
                              connection().lldChannel(),
                              sistrip::extrainfo::scopeModeHeaderHigh_)
                .title();

    highHeader_.histo(dqm()->book1D(title, title, nbins, -0.5, 1024 * 1. - 0.5));
    highHeader_.isProfile_ = false;
    highHeader_.vNumOfEntries_.resize(nbins, 0);

    //// Scope mode histograms
    title = SiStripHistoTitle(sistrip::EXPERT_HISTO,
                              sistrip::DAQ_SCOPE_MODE,
                              sistrip::FED_KEY,
                              fedKey(),
                              sistrip::LLD_CHAN,
                              connection().lldChannel(),
                              sistrip::extrainfo::scopeModeFrame_)
                .title();

    scopeFrame_.histo(dqm()->bookProfile(title, title, nBinsSpy_, -0.5, nBinsSpy_ * 1. - 0.5, 1025, 0., 1025.));

    scopeFrame_.vNumOfEntries_.resize(nBinsSpy_, 0);
    scopeFrame_.vSumOfContents_.resize(nBinsSpy_, 0);
    scopeFrame_.vSumOfSquares_.resize(nBinsSpy_, 0);
    scopeFrame_.isProfile_ = true;
  }
}

// -----------------------------------------------------------------------------
//
void DaqScopeModeTask::fill(const SiStripEventSummary& summary, const edm::DetSet<SiStripRawDigi>& digis) {
  // Only fill every 'N' events
  if (not(parameters_.existsAs<bool>("isSpy") and parameters_.getParameter<bool>("isSpy"))) {
    if (!updateFreq() || fillCntr() % updateFreq()) {
      return;
    }
  }

  if (digis.data.size() != nBins_) {  //@@ check scope mode length?
    edm::LogWarning(mlDqmSource_) << "[DaqScopeModeTask::" << __func__ << "]"
                                  << " Unexpected number of digis (" << digis.data.size()
                                  << ") wrt number of histogram bins (" << nBins_ << ")!";
  }

  if (not(parameters_.existsAs<bool>("isSpy") and parameters_.getParameter<bool>("isSpy"))) {
    uint16_t bins = digis.data.size() < nBins_ ? digis.data.size() : nBins_;
    for (uint16_t ibin = 0; ibin < bins; ibin++) {
      updateHistoSet(scopeFrame_, ibin, digis.data[ibin].adc());
    }
  } else {
    // fill the pedestal histograms as done in the pedestal task
    // Check number of digis
    uint16_t nbins = peds_[0].vNumOfEntries_.size();
    if (digis.data.size() < nbins) {
      nbins = digis.data.size();
    }

    uint16_t napvs = nbins / 128;
    std::vector<uint32_t> cm;
    cm.resize(napvs, 0);

    // Calc common mode for both APVs
    std::vector<uint16_t> adc;
    for (uint16_t iapv = 0; iapv < napvs; iapv++) {
      adc.clear();
      adc.reserve(128);

      for (uint16_t ibin = 0; ibin < 128; ibin++) {
        if ((iapv * 128) + ibin < nbins) {
          adc.push_back(digis.data[(iapv * 128) + ibin].adc());
        }
      }

      sort(adc.begin(), adc.end());
      uint16_t index = adc.size() % 2 ? adc.size() / 2 : adc.size() / 2 - 1;
      if (!adc.empty()) {
        cm[iapv] = static_cast<uint32_t>(adc[index]);
      }
    }

    for (uint16_t ibin = 0; ibin < nbins; ibin++) {
      float digiVal = digis.data[ibin].adc();
      updateHistoSet(peds_[0], ibin, digiVal);  // peds and raw noise
      float diff = digiVal - static_cast<float>(cm[ibin / 128]);
      updateHistoSet(peds_[1], ibin, diff);  // residuals and real noise
    }

    if (cm.size() < cm_.size()) {
      edm::LogWarning(mlDqmSource_) << "[PedestalsTask::" << __func__ << "]"
                                    << " Fewer CM values than expected: " << cm.size();
    }

    updateHistoSet(cm_[0], cm[0]);
    updateHistoSet(cm_[1], cm[1]);
  }
}

// -----------------------------------------------------------------------------
//
void DaqScopeModeTask::fill(const SiStripEventSummary& summary,
                            const edm::DetSet<SiStripRawDigi>& digis,
                            const edm::DetSet<SiStripRawDigi>& digisAlt) {
  // Only fill every 'N' events
  if (not(parameters_.existsAs<bool>("isSpy") and parameters_.getParameter<bool>("isSpy"))) {
    if (!updateFreq() || fillCntr() % updateFreq()) {
      return;
    }
  }

  if (digis.data.size() != nBins_) {  //@@ check scope mode length?
    edm::LogWarning(mlDqmSource_) << "[DaqScopeModeTask::" << __func__ << "]"
                                  << " Unexpected number of digis (" << digis.data.size()
                                  << ") wrt number of histogram bins (" << nBins_ << ")!";
  }

  if (not(parameters_.existsAs<bool>("isSpy") and parameters_.getParameter<bool>("isSpy"))) {
    uint16_t bins = digis.data.size() < nBins_ ? digis.data.size() : nBins_;
    for (uint16_t ibin = 0; ibin < bins; ibin++) {
      updateHistoSet(scopeFrame_, ibin, digis.data[ibin].adc());
    }
  } else {
    // fill the pedestal histograms as done in the pedestal task
    // Check number of digis
    uint16_t nbins = peds_[0].vNumOfEntries_.size();
    if (digis.data.size() < nbins) {
      nbins = digis.data.size();
    }

    uint16_t napvs = nbins / 128;
    std::vector<uint32_t> cm;
    cm.resize(napvs, 0);

    // Calc common mode for both APVs
    std::vector<uint16_t> adc;
    for (uint16_t iapv = 0; iapv < napvs; iapv++) {
      adc.clear();
      adc.reserve(128);
      for (uint16_t ibin = 0; ibin < 128; ibin++) {
        if ((iapv * 128) + ibin < nbins) {
          adc.push_back(digis.data[(iapv * 128) + ibin].adc());
        }
      }
      sort(adc.begin(), adc.end());
      uint16_t index = adc.size() % 2 ? adc.size() / 2 : adc.size() / 2 - 1;
      if (!adc.empty()) {
        cm[iapv] = static_cast<uint32_t>(adc[index]);
      }
    }

    /// Calculate pedestal
    for (uint16_t ibin = 0; ibin < nbins; ibin++) {
      float digiVal = digis.data[ibin].adc();
      updateHistoSet(peds_[0], ibin, digiVal);  // peds and raw noise
      float diff = digiVal - static_cast<float>(cm[ibin / 128]);
      updateHistoSet(peds_[1], ibin, diff);  // residuals and real noise
    }

    if (cm.size() < cm_.size()) {
      edm::LogWarning(mlDqmSource_) << "[PedestalsTask::" << __func__ << "]"
                                    << " Fewer CM values than expected: " << cm.size();
    }

    updateHistoSet(cm_[0], cm[0]);
    updateHistoSet(cm_[1], cm[1]);

    uint16_t bins = digisAlt.data.size() < nBinsSpy_ ? digisAlt.data.size() : nBinsSpy_;
    for (uint16_t ibin = 0; ibin < bins; ibin++) {
      updateHistoSet(scopeFrame_, ibin, digisAlt.data[ibin].adc());
    }

    // Header low and high for both APVs
    std::vector<uint32_t> adcHeader_high;
    std::vector<uint32_t> adcHeader_low;

    float threshold_high = (digisAlt.data[286].adc() + digisAlt.data[287].adc()) / 4;
    float threshold_low = 100;
    int minNumberForHeader = 4;
    bool goodHeaderFound = false;
    int nConsecutiveHigh = 0;
    adcHeader_high.clear();
    adcHeader_high.reserve(30);
    adcHeader_low.clear();
    adcHeader_low.reserve(30);

    for (uint16_t ibin = 6; ibin < 11; ibin++) {
      if (digisAlt.data[ibin].adc() > threshold_high) {
        nConsecutiveHigh++;
      }
    }

    if (nConsecutiveHigh > minNumberForHeader)
      goodHeaderFound = true;  // if nConsecutiveHigh > 4 --> good header found
    if (goodHeaderFound == false)
      return;
    for (uint16_t ibin = 0; ibin < 30; ibin++) {
      if (digisAlt.data[ibin].adc() > threshold_high &&
          goodHeaderFound) {  // save of samples above avg(trailing ticks)/4
        adcHeader_high.push_back(digisAlt.data[ibin].adc());
      }
      if (digisAlt.data[ibin].adc() < threshold_low && goodHeaderFound) {
        adcHeader_low.push_back(digisAlt.data[ibin].adc());
      }
    }
    if (adcHeader_low.empty() || adcHeader_high.empty()) {
      return;
    }
    for (uint16_t i = 0; i < adcHeader_low.size(); i++) {
      updateHistoSet(lowHeader_, adcHeader_low[i]);
    }
    for (uint16_t i = 0; i < adcHeader_high.size(); i++) {
      updateHistoSet(highHeader_, adcHeader_high[i]);
    }
  }
}

// -----------------------------------------------------------------------------
//
void DaqScopeModeTask::fill(const SiStripEventSummary& summary,
                            const edm::DetSet<SiStripRawDigi>& digis,
                            const edm::DetSet<SiStripRawDigi>& digisAlt,
                            const std::vector<uint16_t>& stripOnCluster) {
  // Only fill every 'N' events
  if (not(parameters_.existsAs<bool>("isSpy") and parameters_.getParameter<bool>("isSpy"))) {
    if (!updateFreq() || fillCntr() % updateFreq()) {
      return;
    }
  }

  if (digis.data.size() != nBins_) {  //@@ check scope mode length?
    edm::LogWarning(mlDqmSource_) << "[DaqScopeModeTask::" << __func__ << "]"
                                  << " Unexpected number of digis (" << digis.data.size()
                                  << ") wrt number of histogram bins (" << nBins_ << ")!";
  }

  if (not(parameters_.existsAs<bool>("isSpy") and parameters_.getParameter<bool>("isSpy"))) {
    uint16_t bins = digis.data.size() < nBins_ ? digis.data.size() : nBins_;
    for (uint16_t ibin = 0; ibin < bins; ibin++) {
      updateHistoSet(scopeFrame_, ibin, digis.data[ibin].adc());
    }
  } else {
    // fill the pedestal histograms as done in the pedestal task
    uint16_t nbins = peds_[0].vNumOfEntries_.size();
    if (digis.data.size() < nbins) {
      nbins = digis.data.size();
    }
    uint16_t napvs = nbins / 128;
    std::vector<uint32_t> cm;
    cm.resize(napvs, 0);
    // Calc common mode for both APVs
    std::vector<uint16_t> adc;
    for (uint16_t iapv = 0; iapv < napvs; iapv++) {
      adc.clear();
      adc.reserve(128);
      for (uint16_t ibin = 0; ibin < 128; ibin++) {
        if ((iapv * 128) + ibin < nbins) {
          if (std::find(stripOnCluster.begin(), stripOnCluster.end(), (iapv * 128) + ibin) ==
              stripOnCluster.end())  // if not found, strip is good
            adc.push_back(digis.data[(iapv * 128) + ibin].adc());
        }
      }
      sort(adc.begin(), adc.end());
      uint16_t index = adc.size() % 2 ? adc.size() / 2 : adc.size() / 2 - 1;
      if (!adc.empty()) {
        cm[iapv] = static_cast<uint32_t>(adc[index]);
      }
    }

    /// Calculate pedestal
    for (uint16_t ibin = 0; ibin < nbins; ibin++) {
      if (std::find(stripOnCluster.begin(), stripOnCluster.end(), ibin) != stripOnCluster.end()) {
        continue;
      }
      float digiVal = digis.data[ibin].adc();
      updateHistoSet(peds_[0], ibin, digiVal);  // peds and raw noise
      float diff = digiVal - static_cast<float>(cm[ibin / 128]);
      updateHistoSet(peds_[1], ibin, diff);  // residuals and real noise
    }

    if (cm.size() < cm_.size()) {
      edm::LogWarning(mlDqmSource_) << "[PedestalsTask::" << __func__ << "]"
                                    << " Fewer CM values than expected: " << cm.size();
    }

    updateHistoSet(cm_[0], cm[0]);
    updateHistoSet(cm_[1], cm[1]);

    uint16_t bins = digisAlt.data.size() < nBinsSpy_ ? digisAlt.data.size() : nBinsSpy_;
    for (uint16_t ibin = 0; ibin < bins; ibin++) {
      updateHistoSet(scopeFrame_, ibin, digisAlt.data[ibin].adc());
    }
    // Header low and high for both APVs
    std::vector<uint32_t> adcHeader_high;
    std::vector<uint32_t> adcHeader_low;

    float threshold_high = (digisAlt.data[286].adc() + digisAlt.data[287].adc()) / 4;
    float threshold_low = 120;
    int minNumberForHeader = 4;
    bool goodHeaderFound = false;
    int nConsecutiveHigh = 0;
    adcHeader_high.clear();
    adcHeader_high.reserve(30);
    adcHeader_low.clear();
    adcHeader_low.reserve(30);

    for (uint16_t ibin = 6; ibin < 11; ibin++) {
      if (digisAlt.data[ibin].adc() > threshold_high) {
        nConsecutiveHigh++;
      }
    }

    if (nConsecutiveHigh > minNumberForHeader)
      goodHeaderFound = true;  // if nConsecutiveHigh > 4 --> good header found
    if (goodHeaderFound == false)
      return;
    for (uint16_t ibin = 0; ibin < 30; ibin++) {
      if (digisAlt.data[ibin].adc() > threshold_high &&
          goodHeaderFound) {  // save of samples above avg(trailing ticks)/4
        adcHeader_high.push_back(digisAlt.data[ibin].adc());
      }
      if (digisAlt.data[ibin].adc() < threshold_low && goodHeaderFound) {
        adcHeader_low.push_back(digisAlt.data[ibin].adc());
      }
    }
    if (adcHeader_low.empty() || adcHeader_high.empty()) {
      return;
    }
    for (uint16_t i = 0; i < adcHeader_low.size(); i++) {
      updateHistoSet(lowHeader_, adcHeader_low[i]);
    }
    for (uint16_t i = 0; i < adcHeader_high.size(); i++) {
      updateHistoSet(highHeader_, adcHeader_high[i]);
    }
  }
}

// -----------------------------------------------------------------------------
//
void DaqScopeModeTask::update() {
  if (not(parameters_.existsAs<bool>("isSpy") and parameters_.getParameter<bool>("isSpy")))
    updateHistoSet(scopeFrame_);
  else {
    updateHistoSet(peds_[0]);
    TProfile* histo = ExtractTObject<TProfile>().extract(peds_[1].histo());
    for (uint16_t ii = 0; ii < peds_[1].vNumOfEntries_.size(); ++ii) {
      float mean = 0.;
      float spread = 0.;
      float entries = peds_[1].vNumOfEntries_[ii];
      if (entries > 0.) {
        mean = peds_[1].vSumOfContents_[ii] / entries;
        spread = sqrt(peds_[1].vSumOfSquares_[ii] / entries - mean * mean);
      }

      float noise = spread;
      float error = 0;  // sqrt(entries) / entries;
      UpdateTProfile::setBinContent(histo, ii + 1, entries, noise, error);
    }

    updateHistoSet(cm_[0]);
    updateHistoSet(cm_[1]);

    updateHistoSet(scopeFrame_);
    updateHistoSet(lowHeader_);
    updateHistoSet(highHeader_);
  }
}
