#include "DQM/SiStripCommissioningAnalysis/interface/FedTimingAlgorithm.h"
#include "CondFormats/SiStripObjects/interface/FedTimingAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TProfile.h"
#include "TH1.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace sistrip;

// ----------------------------------------------------------------------------
//
FedTimingAlgorithm::FedTimingAlgorithm(const edm::ParameterSet& pset, FedTimingAnalysis* const anal)
    : CommissioningAlgorithm(anal), histo_(nullptr, "") {
  ;
}

// ----------------------------------------------------------------------------
//
void FedTimingAlgorithm::extract(const std::vector<TH1*>& histos) {
  if (!anal()) {
    edm::LogWarning(mlCommissioning_) << "[FedTimingAlgorithm::" << __func__ << "]"
                                      << " NULL pointer to Analysis object!";
    return;
  }

  // Check number of histograms
  if (histos.size() != 1) {
    anal()->addErrorCode(sistrip::numberOfHistos_);
  }

  // Extract FED key from histo title
  if (!histos.empty()) {
    anal()->fedKey(extractFedKey(histos.front()));
  }

  // Extract histograms
  std::vector<TH1*>::const_iterator ihis = histos.begin();
  for (; ihis != histos.end(); ihis++) {
    // Check for NULL pointer
    if (!(*ihis)) {
      continue;
    }

    // Check name
    SiStripHistoTitle title((*ihis)->GetName());
    if (title.runType() != sistrip::FED_TIMING) {
      anal()->addErrorCode(sistrip::unexpectedTask_);
      continue;
    }

    // Extract timing histo
    histo_.first = *ihis;
    histo_.second = (*ihis)->GetName();
  }
}

// ----------------------------------------------------------------------------
//
void FedTimingAlgorithm::analyse() {
  if (!anal()) {
    edm::LogWarning(mlCommissioning_) << "[FedTimingAlgorithm::" << __func__ << "]"
                                      << " NULL pointer to base Analysis object!";
    return;
  }

  CommissioningAnalysis* tmp = const_cast<CommissioningAnalysis*>(anal());
  FedTimingAnalysis* anal = dynamic_cast<FedTimingAnalysis*>(tmp);
  if (!anal) {
    edm::LogWarning(mlCommissioning_) << "[FedTimingAlgorithm::" << __func__ << "]"
                                      << " NULL pointer to derived Analysis object!";
    return;
  }

  if (!histo_.first) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }

  // Transfer histogram contents/errors/stats to containers
  float max = -1.e9;
  float min = 1.e9;
  uint16_t nbins = static_cast<uint16_t>(histo_.first->GetNbinsX());
  std::vector<float> bin_contents;
  std::vector<float> bin_errors;
  std::vector<float> bin_entries;
  bin_contents.reserve(nbins);
  bin_errors.reserve(nbins);
  bin_entries.reserve(nbins);
  for (uint16_t ibin = 0; ibin < nbins; ibin++) {
    bin_contents.push_back(histo_.first->GetBinContent(ibin + 1));
    bin_errors.push_back(histo_.first->GetBinError(ibin + 1));
    //bin_entries.push_back( histo_.first->GetBinEntries(ibin+1) );
    if (bin_entries[ibin]) {
      if (bin_contents[ibin] > max) {
        max = bin_contents[ibin];
      }
      if (bin_contents[ibin] < min) {
        min = bin_contents[ibin];
      }
    }
  }

  if (bin_contents.size() < 100) {
    anal->addErrorCode(sistrip::numberOfBins_);
    return;
  }

  // Calculate range (max-min) and threshold level (range/2)
  float range = max - min;
  float threshold = min + range / 2.;
  if (range < 50.) {
    anal->addErrorCode(sistrip::smallDataRange_);
    return;
  }
  //LogTrace(mlCommissioning_) << " ADC samples: max/min/range/threshold: "
  //<< max << "/" << min << "/" << range << "/" << threshold;

  // Associate samples with either "tick mark" or "baseline"
  std::vector<float> tick;
  std::vector<float> base;
  for (uint16_t ibin = 0; ibin < nbins; ibin++) {
    if (bin_entries[ibin]) {
      if (bin_contents[ibin] < threshold) {
        base.push_back(bin_contents[ibin]);
      } else {
        tick.push_back(bin_contents[ibin]);
      }
    }
  }
  //LogTrace(mlCommissioning_) << " Number of 'tick mark' samples: " << tick.size()
  //<< " Number of 'baseline' samples: " << base.size();

  // Find median level of tick mark and baseline
  float tickmark = 0.;
  float baseline = 0.;
  sort(tick.begin(), tick.end());
  sort(base.begin(), base.end());
  if (!tick.empty()) {
    tickmark = tick[tick.size() % 2 ? tick.size() / 2 : tick.size() / 2];
  }
  if (!base.empty()) {
    baseline = base[base.size() % 2 ? base.size() / 2 : base.size() / 2];
  }
  //LogTrace(mlCommissioning_) << " Tick mark level: " << tickmark << " Baseline level: " << baseline
  //<< " Range: " << (tickmark-baseline);
  if ((tickmark - baseline) < 50.) {
    anal->addErrorCode(sistrip::smallDataRange_);
    return;
  }

  // Find rms spread in "baseline" samples
  float mean = 0.;
  float mean2 = 0.;
  for (uint16_t ibin = 0; ibin < base.size(); ibin++) {
    mean += base[ibin];
    mean2 += base[ibin] * base[ibin];
  }
  if (!base.empty()) {
    mean = mean / base.size();
    mean2 = mean2 / base.size();
  } else {
    mean = 0.;
    mean2 = 0.;
  }
  float baseline_rms = 0.;
  if (mean2 > mean * mean) {
    baseline_rms = sqrt(mean2 - mean * mean);
  } else {
    baseline_rms = 0.;
  }
  //LogTrace(mlCommissioning_) << " Spread in baseline samples: " << baseline_rms;

  // Find rising edges (derivative across two bins > range/2)
  std::map<uint16_t, float> edges;
  for (uint16_t ibin = 1; ibin < nbins - 1; ibin++) {
    if (bin_entries[ibin + 1] && bin_entries[ibin - 1]) {
      float derivative = bin_contents[ibin + 1] - bin_contents[ibin - 1];
      if (derivative > 5. * baseline_rms) {
        edges[ibin] = derivative;
        //LogTrace(mlCommissioning_) << " Found edge #" << edges.size() << " at bin " << ibin
        //<< " and with derivative " << derivative;
      }
    }
  }

  // Iterate through "edges" std::map
  bool found = false;
  uint16_t deriv_bin = sistrip::invalid_;
  float max_deriv = -1. * sistrip::invalid_;
  std::map<uint16_t, float>::iterator iter = edges.begin();
  while (!found && iter != edges.end()) {
    // Iterate through 50 subsequent samples
    bool valid = true;
    for (uint16_t ii = 0; ii < 50; ii++) {
      uint16_t bin = iter->first + ii;

      // Calc local derivative
      float temp_deriv = 0;
      if (static_cast<uint32_t>(bin) < 1 || static_cast<uint32_t>(bin + 1) >= nbins) {
        continue;
      }
      temp_deriv = bin_contents[bin + 1] - bin_contents[bin - 1];

      // Store max derivative
      if (temp_deriv > max_deriv) {
        max_deriv = temp_deriv;
        deriv_bin = bin;
      }

      // Check if samples following edge are all "high"
      if (ii > 10 && ii < 40 && bin_entries[bin] && bin_contents[bin] < baseline + 5 * baseline_rms) {
        valid = false;
      }
    }

    // Break from loop if tick mark found
    if (valid) {
      found = true;
    } else {
      max_deriv = -1. * sistrip::invalid_;
      deriv_bin = sistrip::invalid_;
      edges.erase(iter);
    }

    iter++;
  }

  // Set monitorables (but not PLL coarse and fine here)
  if (!edges.empty()) {
    anal->time_ = deriv_bin;
    anal->error_ = 0.;
    anal->base_ = baseline;
    anal->peak_ = tickmark;
    anal->height_ = tickmark - baseline;
  } else {
    anal->addErrorCode(sistrip::missingTickMark_);
    anal->base_ = baseline;
    anal->peak_ = tickmark;
    anal->height_ = tickmark - baseline;
  }
}
