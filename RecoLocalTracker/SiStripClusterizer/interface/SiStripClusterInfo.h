#ifndef SISTRIPCLUSTERIZER_SISTRIPCLUSTERINFO_H
#define SISTRIPCLUSTERIZER_SISTRIPCLUSTERINFO_H

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

class SiStripClusterInfo {
public:
  SiStripClusterInfo(edm::ConsumesCollector&&, const std::string& qualityLabel = "");

  void initEvent(const edm::EventSetup& iSetup);
  void setCluster(const SiStripCluster& cluster, int detId);

  const SiStripCluster* cluster() const { return cluster_ptr; }

  uint32_t detId() const { return detId_; }
  uint16_t width() const { return cluster()->amplitudes().size(); }
  uint16_t firstStrip() const { return cluster()->firstStrip(); }
  float baryStrip() const { return cluster()->barycenter(); }
  uint16_t maxStrip() const { return firstStrip() + maxIndex(); }
  float variance() const;

  auto stripCharges() const -> decltype(cluster()->amplitudes()) { return cluster()->amplitudes(); }
  std::vector<float> stripGains() const;
  std::vector<float> stripNoises() const;
  std::vector<float> stripNoisesRescaledByGain() const;
  std::vector<bool> stripQualitiesBad() const;

  uint16_t charge() const { return std::accumulate(stripCharges().begin(), stripCharges().end(), uint16_t(0)); }
  uint8_t maxCharge() const { return *std::max_element(stripCharges().begin(), stripCharges().end()); }
  uint16_t maxIndex() const {
    return std::max_element(stripCharges().begin(), stripCharges().end()) - stripCharges().begin();
  }
  std::pair<uint16_t, uint16_t> chargeLR() const;

  float noise() const { return calculate_noise(stripNoises()); }
  float noiseRescaledByGain() const { return calculate_noise(stripNoisesRescaledByGain()); }

  float signalOverNoise() const { return charge() / noiseRescaledByGain(); }

  bool IsAnythingBad() const;
  bool IsApvBad() const;
  bool IsFiberBad() const;
  bool IsModuleBad() const;
  bool IsModuleUsable() const;

  const SiStripGain* siStripGain() const { return siStripGain_; }
  const SiStripQuality* siStripQuality() const { return siStripQuality_; }

private:
  float calculate_noise(const std::vector<float>&) const;

  const SiStripCluster* cluster_ptr = nullptr;

  edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> siStripNoisesToken_;
  edm::ESGetToken<SiStripGain, SiStripGainRcd> siStripGainToken_;
  edm::ESGetToken<SiStripQuality, SiStripQualityRcd> siStripQualityToken_;

  const SiStripNoises* siStripNoises_ = nullptr;
  const SiStripGain* siStripGain_ = nullptr;
  const SiStripQuality* siStripQuality_ = nullptr;

  uint32_t detId_ = 0;
};

#endif
