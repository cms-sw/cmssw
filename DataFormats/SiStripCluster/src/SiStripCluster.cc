
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

SiStripCluster::SiStripCluster(const SiStripDigiRange& range) : firstStrip_(range.first->strip()), error_x(-99999.9) {
  std::vector<uint8_t> v;
  v.reserve(range.second - range.first);

  uint16_t lastStrip = 0;
  bool firstInloop = true;
  for (SiStripDigiIter i = range.first; i != range.second; i++) {
    /// check if digis consecutive
    if (!firstInloop && i->strip() != lastStrip + 1) {
      for (int j = 0; j < i->strip() - (lastStrip + 1); j++) {
        v.push_back(0);
      }
    }
    lastStrip = i->strip();
    firstInloop = false;

    v.push_back(i->adc());
  }
  amplitudes_ = v;
}

float SiStripCluster::barycenter() const {
  int sumx = 0;
  int suma = 0;
  auto asize = size();
  for (auto i = 0U; i < asize; ++i) {
    sumx += i * amplitudes_[i];
    suma += amplitudes_[i];
  }

  // strip centers are offcet by half pitch w.r.t. strip numbers,
  // so one has to add 0.5 to get the correct barycenter position.
  // Need to mask off the high bit of firstStrip_, which contains the merged status.
  return float((firstStrip_ & stripIndexMask)) + float(sumx) / float(suma) + 0.5f;
}
