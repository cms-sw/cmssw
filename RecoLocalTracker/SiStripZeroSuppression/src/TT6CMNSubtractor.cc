#include "RecoLocalTracker/SiStripZeroSuppression/interface/TT6CMNSubtractor.h"

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include <cmath>

void TT6CMNSubtractor::init(const edm::EventSetup& es) {
  if (noiseWatcher_.check(es)) {
    noiseHandle = &es.getData(noiseToken_);
  }
  if (qualityWatcher_.check(es)) {
    qualityHandle = &es.getData(qualityToken_);
  }
}

void TT6CMNSubtractor::subtract(uint32_t detId, uint16_t firstAPV, std::vector<int16_t>& digis) {
  subtract_(detId, firstAPV, digis);
}
void TT6CMNSubtractor::subtract(uint32_t detId, uint16_t firstAPV, std::vector<float>& digis) {
  subtract_(detId, firstAPV, digis);
}

template <typename T>
inline void TT6CMNSubtractor::subtract_(uint32_t detId, uint16_t firstAPV, std::vector<T>& digis) {
  short FixedBias = 128;
  SiStripNoises::Range detNoiseRange = noiseHandle->getRange(detId);
  SiStripQuality::Range detQualityRange = qualityHandle->getRange(detId);

  typename std::vector<T>::iterator fs, ls;
  double sumVal = 0.0;
  double sumWt = 0.0;

  for (uint16_t istrip = 0; istrip < digis.size(); ++istrip) {
    if (!qualityHandle->IsStripBad(detQualityRange, istrip + firstAPV * 128)) {
      float stripNoise = noiseHandle->getNoiseFast(istrip + firstAPV * 128, detNoiseRange);

      if (std::abs(digis[istrip] - FixedBias) < cut_to_avoid_signal_ * stripNoise) {
        double nWeight = 1 / (stripNoise * stripNoise);
        sumVal += (digis[istrip] - FixedBias) * nWeight;
        sumWt += nWeight;
      }
    }

    if (istrip % 128 == 127) {
      double CM = (sumWt) ? sumVal / sumWt : 0.0;

      fs = digis.begin() + istrip - 127;
      ls = digis.begin() + istrip + 1;

      while (fs < ls) {
        *fs = static_cast<T>(*fs - FixedBias - CM);
        fs++;
      }

      sumVal = 0.0;
      sumWt = 0.0;
    }
  }
}
