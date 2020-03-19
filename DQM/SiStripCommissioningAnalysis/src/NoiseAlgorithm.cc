#include "DQM/SiStripCommissioningAnalysis/interface/NoiseAlgorithm.h"
#include "CondFormats/SiStripObjects/interface/NoiseAnalysis.h"
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
NoiseAlgorithm::NoiseAlgorithm(const edm::ParameterSet& pset, NoiseAnalysis* const anal)
    : CommissioningAlgorithm(anal), hPeds_(nullptr, ""), hNoise_(nullptr, "") {
  ;
}

// ----------------------------------------------------------------------------
//
void NoiseAlgorithm::extract(const std::vector<TH1*>& histos) {
  if (!anal()) {
    edm::LogWarning(mlCommissioning_) << "[NoiseAlgorithm::" << __func__ << "]"
                                      << " NULL pointer to Analysis object!";
    return;
  }

  // Check number of histograms
  if (histos.size() != 2) {
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

    // Check run type
    SiStripHistoTitle title((*ihis)->GetName());
    if (title.runType() != sistrip::NOISE) {
      anal()->addErrorCode(sistrip::unexpectedTask_);
      continue;
    }

    // Extract peds and noise histos (check for legacy names first!)
    if (title.extraInfo().find(sistrip::extrainfo::pedsAndRawNoise_) != std::string::npos) {
      hPeds_.first = *ihis;
      hPeds_.second = (*ihis)->GetName();
      NoiseAnalysis* a = dynamic_cast<NoiseAnalysis*>(const_cast<CommissioningAnalysis*>(anal()));
      if (a) {
        a->legacy_ = true;
      }
    } else if (title.extraInfo().find(sistrip::extrainfo::pedsAndCmSubNoise_) != std::string::npos) {
      hNoise_.first = *ihis;
      hNoise_.second = (*ihis)->GetName();
      NoiseAnalysis* a = dynamic_cast<NoiseAnalysis*>(const_cast<CommissioningAnalysis*>(anal()));
      if (a) {
        a->legacy_ = true;
      }
    } else if (title.extraInfo().find(sistrip::extrainfo::pedestals_) != std::string::npos) {
      hPeds_.first = *ihis;
      hPeds_.second = (*ihis)->GetName();
    } else if (title.extraInfo().find(sistrip::extrainfo::noise_) != std::string::npos) {
      hNoise_.first = *ihis;
      hNoise_.second = (*ihis)->GetName();
    } else if (title.extraInfo().find(sistrip::extrainfo::commonMode_) != std::string::npos) {
      //@@ something here for CM plots?
    } else {
      anal()->addErrorCode(sistrip::unexpectedExtraInfo_);
    }
  }
}

// -----------------------------------------------------------------------------
//
void NoiseAlgorithm::analyse() {
  if (!anal()) {
    edm::LogWarning(mlCommissioning_) << "[NoiseAlgorithm::" << __func__ << "]"
                                      << " NULL pointer to base Analysis object!";
    return;
  }

  CommissioningAnalysis* tmp = const_cast<CommissioningAnalysis*>(anal());
  NoiseAnalysis* anal = dynamic_cast<NoiseAnalysis*>(tmp);
  if (!anal) {
    edm::LogWarning(mlCommissioning_) << "[NoiseAlgorithm::" << __func__ << "]"
                                      << " NULL pointer to derived Analysis object!";
    return;
  }

  if (!hPeds_.first) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }

  if (!hNoise_.first) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }

  TProfile* peds_histo = dynamic_cast<TProfile*>(hPeds_.first);
  TProfile* noise_histo = dynamic_cast<TProfile*>(hNoise_.first);

  if (!peds_histo) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }

  if (!noise_histo) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }

  if (peds_histo->GetNbinsX() != 256) {
    anal->addErrorCode(sistrip::numberOfBins_);
    return;
  }

  if (noise_histo->GetNbinsX() != 256) {
    anal->addErrorCode(sistrip::numberOfBins_);
    return;
  }

  // Iterate through APVs
  for (uint16_t iapv = 0; iapv < 2; iapv++) {
    // Used to calc mean and rms for peds and noise
    float p_sum = 0., p_sum2 = 0., p_max = -1. * sistrip::invalid_, p_min = sistrip::invalid_;
    float n_sum = 0., n_sum2 = 0., n_max = -1. * sistrip::invalid_, n_min = sistrip::invalid_;
    float r_sum = 0., r_sum2 = 0., r_max = -1. * sistrip::invalid_, r_min = sistrip::invalid_;

    // Iterate through strips of APV
    for (uint16_t istr = 0; istr < 128; istr++) {
      uint16_t strip = iapv * 128 + istr;

      // Pedestals and raw noise
      if (peds_histo) {
        if (peds_histo->GetBinEntries(strip + 1)) {
          anal->peds_[iapv][istr] = peds_histo->GetBinContent(strip + 1);
          p_sum += anal->peds_[iapv][istr];
          p_sum2 += (anal->peds_[iapv][istr] * anal->peds_[iapv][istr]);
          if (anal->peds_[iapv][istr] > p_max) {
            p_max = anal->peds_[iapv][istr];
          }
          if (anal->peds_[iapv][istr] < p_min) {
            p_min = anal->peds_[iapv][istr];
          }

          anal->raw_[iapv][istr] = peds_histo->GetBinError(strip + 1);
          r_sum += anal->raw_[iapv][istr];
          r_sum2 += (anal->raw_[iapv][istr] * anal->raw_[iapv][istr]);
          if (anal->raw_[iapv][istr] > r_max) {
            r_max = anal->raw_[iapv][istr];
          }
          if (anal->raw_[iapv][istr] < r_min) {
            r_min = anal->raw_[iapv][istr];
          }
        }
      }

      // Noise
      if (noise_histo) {
        if (noise_histo->GetBinEntries(strip + 1)) {
          anal->noise_[iapv][istr] = noise_histo->GetBinContent(strip + 1);
          n_sum += anal->noise_[iapv][istr];
          n_sum2 += (anal->noise_[iapv][istr] * anal->noise_[iapv][istr]);
          if (anal->noise_[iapv][istr] > n_max) {
            n_max = anal->noise_[iapv][istr];
          }
          if (anal->noise_[iapv][istr] < n_min) {
            n_min = anal->noise_[iapv][istr];
          }
        }
      }

    }  // strip loop

    // Calc mean and rms for peds
    if (!anal->peds_[iapv].empty()) {
      p_sum /= static_cast<float>(anal->peds_[iapv].size());
      p_sum2 /= static_cast<float>(anal->peds_[iapv].size());
      anal->pedsMean_[iapv] = p_sum;
      anal->pedsSpread_[iapv] = sqrt(fabs(p_sum2 - p_sum * p_sum));
    }

    // Calc mean and rms for noise
    if (!anal->noise_[iapv].empty()) {
      n_sum /= static_cast<float>(anal->noise_[iapv].size());
      n_sum2 /= static_cast<float>(anal->noise_[iapv].size());
      anal->noiseMean_[iapv] = n_sum;
      anal->noiseSpread_[iapv] = sqrt(fabs(n_sum2 - n_sum * n_sum));
    }

    // Calc mean and rms for raw noise
    if (!anal->raw_[iapv].empty()) {
      r_sum /= static_cast<float>(anal->raw_[iapv].size());
      r_sum2 /= static_cast<float>(anal->raw_[iapv].size());
      anal->rawMean_[iapv] = r_sum;
      anal->rawSpread_[iapv] = sqrt(fabs(r_sum2 - r_sum * r_sum));
    }

    // Set max and min values for peds, noise and raw noise
    if (p_max > -1. * sistrip::maximum_) {
      anal->pedsMax_[iapv] = p_max;
    }
    if (p_min < 1. * sistrip::maximum_) {
      anal->pedsMin_[iapv] = p_min;
    }
    if (n_max > -1. * sistrip::maximum_) {
      anal->noiseMax_[iapv] = n_max;
    }
    if (n_min < 1. * sistrip::maximum_) {
      anal->noiseMin_[iapv] = n_min;
    }
    if (r_max > -1. * sistrip::maximum_) {
      anal->rawMax_[iapv] = r_max;
    }
    if (r_min < 1. * sistrip::maximum_) {
      anal->rawMin_[iapv] = r_min;
    }

    // Set dead and noisy strips
    for (uint16_t istr = 0; istr < 128; istr++) {
      if (anal->noiseMin_[iapv] > sistrip::maximum_ || anal->noiseMax_[iapv] > sistrip::maximum_) {
        continue;
      }
      if (anal->noise_[iapv][istr] < (anal->noiseMean_[iapv] - 5. * anal->noiseSpread_[iapv])) {
        anal->dead_[iapv].push_back(istr);  //@@ valid threshold???
      } else if (anal->noise_[iapv][istr] > (anal->noiseMean_[iapv] + 5. * anal->noiseSpread_[iapv])) {
        anal->noisy_[iapv].push_back(istr);  //@@ valid threshold???
      }
    }

  }  // apv loop
}
