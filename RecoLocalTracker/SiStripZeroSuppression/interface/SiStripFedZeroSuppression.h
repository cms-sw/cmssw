#ifndef RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPZEROSUPPRESSOR_H
#define RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPZEROSUPPRESSOR_H

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <vector>
class SiStripNoises;
class SiStripThreshold;

class SiStripFedZeroSuppression {
  friend class SiStripRawProcessingFactory;

public:
  SiStripFedZeroSuppression(uint16_t fedalgo, bool trunc = true, bool trunc10bits = false)
      : noise_cache_id(0),
        threshold_cache_id(0),
        theFEDalgorithm(fedalgo),
        doTruncate(trunc),
        doTruncate10bits(trunc10bits) {}
  ~SiStripFedZeroSuppression(){};
  void init(const edm::EventSetup& es);
  void suppress(const std::vector<SiStripDigi>& in,
                std::vector<SiStripDigi>& selectedSignal,
                uint32_t detId,
                edm::ESHandle<SiStripNoises>&,
                edm::ESHandle<SiStripThreshold>&);
  void suppress(const std::vector<SiStripDigi>& in, std::vector<SiStripDigi>& selectedSignal, uint32_t detId);
  void suppress(const edm::DetSet<SiStripRawDigi>& in, edm::DetSet<SiStripDigi>& out);
  void suppress(const std::vector<int16_t>& in, uint16_t firstAPV, edm::DetSet<SiStripDigi>& out);

  uint16_t truncate(int16_t adc) const {
    if (adc > 253 && doTruncate && !doTruncate10bits)
      return ((adc == 1023) ? 255 : 254);
    return adc;
  };

private:
  edm::ESHandle<SiStripNoises> noiseHandle;
  edm::ESHandle<SiStripThreshold> thresholdHandle;
  uint32_t noise_cache_id, threshold_cache_id;

  uint16_t theFEDalgorithm;
  bool isAValidDigi();

  bool doTruncate;
  bool doTruncate10bits;
  int16_t theFEDlowThresh;
  int16_t theFEDhighThresh;

  int16_t adc;
  int16_t adcPrev;
  int16_t adcNext;
  int16_t adcMaxNeigh;
  int16_t adcPrev2;
  int16_t adcNext2;

  int16_t thePrevFEDlowThresh;
  int16_t thePrevFEDhighThresh;
  int16_t theNextFEDlowThresh;
  int16_t theNextFEDhighThresh;

  int16_t theNeighFEDlowThresh;
  int16_t theNeighFEDhighThresh;

  int16_t thePrev2FEDlowThresh;
  int16_t theNext2FEDlowThresh;

  // working caches
  std::vector<int16_t> highThr_, lowThr_;    // thresholds in adc counts
  std::vector<float> highThrSN_, lowThrSN_;  // thresholds as S/N
  std::vector<float> noises_;

  void fillThresholds_(const uint32_t detID, size_t size);
};
#endif
