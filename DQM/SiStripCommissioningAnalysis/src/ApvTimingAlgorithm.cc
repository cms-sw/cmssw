#include "DQM/SiStripCommissioningAnalysis/interface/ApvTimingAlgorithm.h"
#include "CondFormats/SiStripObjects/interface/ApvTimingAnalysis.h"
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
ApvTimingAlgorithm::ApvTimingAlgorithm(const edm::ParameterSet& pset, ApvTimingAnalysis* const anal)
    : CommissioningAlgorithm(anal), histo_(nullptr, "") {
  ;
}

// ----------------------------------------------------------------------------
//
void ApvTimingAlgorithm::extract(const std::vector<TH1*>& histos) {
  if (!anal()) {
    edm::LogWarning(mlCommissioning_) << "[ApvTimingAlgorithm::" << __func__ << "]"
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
    if (title.runType() != sistrip::APV_TIMING) {
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
void ApvTimingAlgorithm::analyse() {
  if (!anal()) {
    edm::LogWarning(mlCommissioning_) << "[ApvTimingAlgorithm::" << __func__ << "]"
                                      << " NULL pointer to base Analysis object!";
    return;
  }

  CommissioningAnalysis* tmp = const_cast<CommissioningAnalysis*>(anal());
  ApvTimingAnalysis* anal = dynamic_cast<ApvTimingAnalysis*>(tmp);
  if (!anal) {
    edm::LogWarning(mlCommissioning_) << "[ApvTimingAlgorithm::" << __func__ << "]"
                                      << " NULL pointer to derived Analysis object!";
    return;
  }

  if (!histo_.first) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }

  TProfile* histo = dynamic_cast<TProfile*>(histo_.first);
  if (!histo) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }

  // Transfer histogram contents/errors/stats to containers
  float max = -1. * sistrip::invalid_;
  float min = 1. * sistrip::invalid_;
  uint16_t nbins = static_cast<uint16_t>(histo->GetNbinsX());
  std::vector<float> bin_contents;
  std::vector<float> bin_errors;
  std::vector<float> bin_entries;
  bin_contents.reserve(nbins);
  bin_errors.reserve(nbins);
  bin_entries.reserve(nbins);
  for (uint16_t ibin = 0; ibin < nbins; ibin++) {
    bin_contents.push_back(histo->GetBinContent(ibin + 1));
    bin_errors.push_back(histo->GetBinError(ibin + 1));
    bin_entries.push_back(histo->GetBinEntries(ibin + 1));
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
  float threshold = min + (max - min) / 2.;
  anal->base_ = min;
  anal->peak_ = max;
  anal->height_ = max - min;
  if (max - min < ApvTimingAnalysis::tickMarkHeightThreshold_) {
    anal->addErrorCode(sistrip::smallDataRange_);
    return;
  }

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
  anal->base_ = baseline;
  anal->peak_ = tickmark;
  anal->height_ = tickmark - baseline;
  if (tickmark - baseline < ApvTimingAnalysis::tickMarkHeightThreshold_) {
    anal->addErrorCode(sistrip::smallTickMarkHeight_);
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
  float baseline_rms = sqrt(fabs(mean2 - mean * mean));

  // Find rising edges (derivative across two bins > threshold)
  std::map<uint16_t, float> edges;
  for (uint16_t ibin = 1; ibin < nbins - 1; ibin++) {
    if (bin_entries[ibin + 1] && bin_entries[ibin - 1]) {
      float derivative = bin_contents[ibin + 1] - bin_contents[ibin - 1];
      if (derivative > 3. * baseline_rms) {
        edges[ibin] = derivative;
      }
    }
  }
  if (edges.empty()) {
    anal->addErrorCode(sistrip::noRisingEdges_);
    return;
  }

  // Iterate through "edges" map
  uint16_t max_derivative_bin = sistrip::invalid_;
  float max_derivative = -1. * sistrip::invalid_;

  bool found = false;
  std::map<uint16_t, float>::iterator iter = edges.begin();
  while (!found && iter != edges.end()) {
    // Iterate through 50 subsequent samples
    bool valid = true;
    for (uint16_t ii = 0; ii < 50; ii++) {
      uint16_t bin = iter->first + ii;

      // Calc local derivative
      float temp = 0.;
      if (static_cast<uint32_t>(bin) < 1 || static_cast<uint32_t>(bin + 1) >= nbins) {
        valid = false;  //@@ require complete plateau is found within histo
        anal->addErrorCode(sistrip::incompletePlateau_);
        continue;
      }
      temp = bin_contents[bin + 1] - bin_contents[bin - 1];

      // Store max derivative
      if (temp > max_derivative) {
        max_derivative = temp;
        max_derivative_bin = bin;
      }

      // Check if samples following edge are all "high"
      if (ii > 10 && ii < 40 && bin_entries[bin] && bin_contents[bin] < baseline + 5. * baseline_rms) {
        valid = false;
      }
    }

    // Break from loop if tick mark found
    if (valid) {
      found = true;
    }

    /*
    else {
      max_derivative = -1.*sistrip::invalid_;
      max_derivative_bin = sistrip::invalid_;
      //edges.erase(iter);
      anal->addErrorCode(sistrip::rejectedCandidate_);
    }
    */

    iter++;  // next candidate
  }

  if (!found) {  //Try tick mark recovery

    max_derivative_bin = sistrip::invalid_;
    max_derivative = -1. * sistrip::invalid_;

    // Find rising edges_r (derivative_r across five bins > threshold)
    std::map<uint16_t, float> edges_r;
    for (uint16_t ibin_r = 1; ibin_r < nbins - 1; ibin_r++) {
      if (bin_entries[ibin_r + 4] && bin_entries[ibin_r + 3] && bin_entries[ibin_r + 2] && bin_entries[ibin_r + 1] &&
          bin_entries[ibin_r] && bin_entries[ibin_r - 1]) {
        float derivative_r = bin_contents[ibin_r + 1] - bin_contents[ibin_r - 1];
        float derivative_r1 = bin_contents[ibin_r + 1] - bin_contents[ibin_r];
        float derivative_r2 = bin_contents[ibin_r + 2] - bin_contents[ibin_r + 1];
        float derivative_r3 = bin_contents[ibin_r + 3] - bin_contents[ibin_r + 2];

        if (derivative_r > 3. * baseline_rms && derivative_r1 > 1. * baseline_rms &&
            derivative_r2 > 1. * baseline_rms && derivative_r3 > 1. * baseline_rms) {
          edges_r[ibin_r] = derivative_r;
        }
      }
    }
    if (edges_r.empty()) {
      anal->addErrorCode(sistrip::noRisingEdges_);
      return;
    }

    // Iterate through "edges_r" map
    float max_derivative_r = -1. * sistrip::invalid_;

    bool found_r = false;
    std::map<uint16_t, float>::iterator iter_r = edges_r.begin();
    while (!found_r && iter_r != edges_r.end()) {
      // Iterate through 50 subsequent samples
      bool valid_r = true;
      int lowpointcount_r = 0;
      const int lowpointallow_r = 25;  //Number of points allowed to fall below threshhold w/o invalidating tick mark
      for (uint16_t ii_r = 0; ii_r < 50; ii_r++) {
        uint16_t bin_r = iter_r->first + ii_r;

        // Calc local derivative_r
        float temp_r = 0.;
        if (static_cast<uint32_t>(bin_r) < 1 || static_cast<uint32_t>(bin_r + 1) >= nbins) {
          valid_r = false;  //@@ require complete plateau is found_r within histo
          anal->addErrorCode(sistrip::incompletePlateau_);
          continue;
        }
        temp_r = bin_contents[bin_r + 1] - bin_contents[bin_r - 1];

        // Store max derivative_r
        if (temp_r > max_derivative_r && ii_r < 10) {
          max_derivative_r = temp_r;
          max_derivative = temp_r;
          max_derivative_bin = bin_r;
        }

        // Check if majority of samples following edge are all "high"
        if (ii_r > 10 && ii_r < 40 && bin_entries[bin_r] && bin_contents[bin_r] < baseline + 5. * baseline_rms) {
          lowpointcount_r++;
          if (lowpointcount_r > lowpointallow_r) {
            valid_r = false;
          }
        }
      }

      // Break from loop if recovery tick mark found
      if (valid_r) {
        found_r = true;
        found = true;
        anal->addErrorCode(sistrip::tickMarkRecovered_);
      } else {
        max_derivative_r = -1. * sistrip::invalid_;
        max_derivative = -1. * sistrip::invalid_;
        max_derivative_bin = sistrip::invalid_;
        //edges_r.erase(iter_r);
        anal->addErrorCode(sistrip::rejectedCandidate_);
      }

      iter_r++;  // next candidate
    }
  }  //End tick mark recovery

  // Record time monitorable and check tick mark height
  if (max_derivative_bin <= sistrip::valid_) {
    anal->time_ = max_derivative_bin * 25. / 24.;
    if (anal->height_ < ApvTimingAnalysis::tickMarkHeightThreshold_) {
      anal->addErrorCode(sistrip::tickMarkBelowThresh_);
    }
  } else {
    anal->addErrorCode(sistrip::missingTickMark_);
  }
}
