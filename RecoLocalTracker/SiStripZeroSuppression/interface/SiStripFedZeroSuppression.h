#ifndef RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPZEROSUPPRESSOR_H
#define RECOLOCALTRACKER_SISTRIPZEROSUPPRESSION_SISTRIPZEROSUPPRESSOR_H

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/DataRecord/interface/SiStripThresholdRcd.h"

#include <vector>
class SiStripNoises;
class SiStripThreshold;

class SiStripFedZeroSuppression {
  friend class SiStripRawProcessingFactory;

public:
  SiStripFedZeroSuppression(uint16_t fedalgo,
                            edm::ConsumesCollector* iC = nullptr,
                            bool trunc = true,
                            bool trunc10bits = false)
      : noiseToken_{iC ? decltype(noiseToken_){iC->esConsumes<SiStripNoises, SiStripNoisesRcd>()}
                       : decltype(noiseToken_){}},
        thresholdToken_{iC ? decltype(thresholdToken_){iC->esConsumes<SiStripThreshold, SiStripThresholdRcd>()}
                           : decltype(thresholdToken_){}},
        theFEDalgorithm(fedalgo),
        doTruncate(trunc),
        doTruncate10bits(trunc10bits) {}
  ~SiStripFedZeroSuppression(){};
  void init(const edm::EventSetup& es);
  void suppress(const std::vector<SiStripDigi>& in,
                std::vector<SiStripDigi>& selectedSignal,
                uint32_t detId,
                const SiStripNoises&,
                const SiStripThreshold&);
  void suppress(const std::vector<SiStripDigi>& in, std::vector<SiStripDigi>& selectedSignal, uint32_t detId);
  void suppress(const edm::DetSet<SiStripRawDigi>& in, edm::DetSet<SiStripDigi>& out);
  void suppress(const std::vector<int16_t>& in, uint16_t firstAPV, edm::DetSet<SiStripDigi>& out);

  uint16_t truncate(int16_t adc) const {
    if (adc > 253 && doTruncate && !doTruncate10bits)
      return ((adc == 1023) ? 255 : 254);
    return adc;
  };

private:
  edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> noiseToken_;
  edm::ESGetToken<SiStripThreshold, SiStripThresholdRcd> thresholdToken_;
  const SiStripNoises* noise_;
  const SiStripThreshold* threshold_;
  edm::ESWatcher<SiStripNoisesRcd> noiseWatcher_;
  edm::ESWatcher<SiStripThresholdRcd> thresholdWatcher_;

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
