#include "DataFormats/EcalDigi/interface/EcalTimeDigi.h"

namespace {
  constexpr unsigned int MAXSAMPLES = 10;
}  // namespace

EcalTimeDigi::EcalTimeDigi() : id_(0), size_(0), sampleOfInterest_(-1), waveform_(WAVEFORMSAMPLES), data_(MAXSAMPLES) {}

EcalTimeDigi::EcalTimeDigi(const DetId& id)
    : id_(id), size_(0), sampleOfInterest_(-1), waveform_(WAVEFORMSAMPLES), data_(MAXSAMPLES) {}

void EcalTimeDigi::setSize(unsigned int size) {
  if (size > MAXSAMPLES)
    size_ = MAXSAMPLES;
  else
    size_ = size;
  data_.resize(size_);
}

void EcalTimeDigi::setWaveform(float* waveform) {
  waveform_.resize(WAVEFORMSAMPLES);
  for (uint i(0); i != WAVEFORMSAMPLES; ++i) {
    waveform_[i] = waveform[i];
  }
}
