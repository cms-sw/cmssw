#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterInfo.h"

#include <cmath>

SiStripClusterInfo::SiStripClusterInfo(edm::ConsumesCollector&& iC, const std::string& qualityLabel)
    : siStripNoisesToken_{iC.esConsumes<SiStripNoises, SiStripNoisesRcd>()},
      siStripGainToken_{iC.esConsumes<SiStripGain, SiStripGainRcd>()},
      siStripQualityToken_{iC.esConsumes<SiStripQuality, SiStripQualityRcd>(edm::ESInputTag("", qualityLabel))} {}

void SiStripClusterInfo::initEvent(const edm::EventSetup& iSetup) {
  siStripNoises_ = &iSetup.getData(siStripNoisesToken_);
  siStripGain_ = &iSetup.getData(siStripGainToken_);
  siStripQuality_ = &iSetup.getData(siStripQualityToken_);
}

void SiStripClusterInfo::setCluster(const SiStripCluster& cluster, int detId) {
  cluster_ptr = &cluster;
  detId_ = detId;
}

std::pair<uint16_t, uint16_t> SiStripClusterInfo::chargeLR() const {
  std::vector<uint8_t>::const_iterator begin(stripCharges().begin()), end(stripCharges().end()), max;
  max = max_element(begin, end);
  return std::make_pair(accumulate(begin, max, uint16_t(0)), accumulate(max + 1, end, uint16_t(0)));
}

float SiStripClusterInfo::variance() const {
  float q(0), x1(0), x2(0);
  for (auto begin(stripCharges().begin()), end(stripCharges().end()), it(begin); it != end; ++it) {
    unsigned i = it - begin;
    q += (*it);
    x1 += (*it) * (i + 0.5);
    x2 += (*it) * (i * i + i + 1. / 3);
  }
  return (x2 - x1 * x1 / q) / q;
}

std::vector<float> SiStripClusterInfo::stripNoisesRescaledByGain() const {
  SiStripNoises::Range detNoiseRange = siStripNoises_->getRange(detId_);
  SiStripApvGain::Range detGainRange = siStripGain_->getRange(detId_);

  std::vector<float> results;
  results.reserve(width());
  for (size_t i = 0, e = width(); i < e; i++) {
    results.push_back(siStripNoises_->getNoise(firstStrip() + i, detNoiseRange) /
                      siStripGain_->getStripGain(firstStrip() + i, detGainRange));
  }
  return results;
}

std::vector<float> SiStripClusterInfo::stripNoises() const {
  SiStripNoises::Range detNoiseRange = siStripNoises_->getRange(detId_);

  std::vector<float> noises;
  noises.reserve(width());
  for (size_t i = 0; i < width(); i++) {
    noises.push_back(siStripNoises_->getNoise(firstStrip() + i, detNoiseRange));
  }
  return noises;
}

std::vector<float> SiStripClusterInfo::stripGains() const {
  SiStripApvGain::Range detGainRange = siStripGain_->getRange(detId_);

  std::vector<float> gains;
  gains.reserve(width());
  for (size_t i = 0; i < width(); i++) {
    gains.push_back(siStripGain_->getStripGain(firstStrip() + i, detGainRange));
  }
  return gains;
}

std::vector<bool> SiStripClusterInfo::stripQualitiesBad() const {
  std::vector<bool> isBad;
  isBad.reserve(width());
  for (int i = 0; i < width(); i++) {
    isBad.push_back(siStripQuality_->IsStripBad(detId_, firstStrip() + i));
  }
  return isBad;
}

float SiStripClusterInfo::calculate_noise(const std::vector<float>& noise) const {
  float noiseSumInQuadrature = 0;
  int numberStripsOverThreshold = 0;
  for (int i = 0; i < width(); i++) {
    if (stripCharges()[i] != 0) {
      noiseSumInQuadrature += noise.at(i) * noise.at(i);
      numberStripsOverThreshold++;
    }
  }
  return std::sqrt(noiseSumInQuadrature / numberStripsOverThreshold);
}

bool SiStripClusterInfo::IsAnythingBad() const {
  std::vector<bool> stripBad = stripQualitiesBad();
  return IsApvBad() || IsFiberBad() || IsModuleBad() ||
         accumulate(stripBad.begin(), stripBad.end(), false, std::logical_or<bool>());
}

bool SiStripClusterInfo::IsApvBad() const {
  return siStripQuality_->IsApvBad(detId_, firstStrip() / 128) ||
         siStripQuality_->IsApvBad(detId_, (firstStrip() + width()) / 128);
}

bool SiStripClusterInfo::IsFiberBad() const {
  return siStripQuality_->IsFiberBad(detId_, firstStrip() / 256) ||
         siStripQuality_->IsFiberBad(detId_, (firstStrip() + width()) / 256);
}

bool SiStripClusterInfo::IsModuleBad() const { return siStripQuality_->IsModuleBad(detId_); }

bool SiStripClusterInfo::IsModuleUsable() const { return siStripQuality_->IsModuleUsable(detId_); }
