#include "FWCore/Utilities/interface/Likely.h"
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
  initQB();
}

SiStripCluster::SiStripCluster(const SiStripApproximateCluster cluster, const uint16_t maxStrips) : error_x(-99999.9) {
  barycenter_ = cluster.barycenter() / 10.0;
  charge_ = cluster.width() * cluster.avgCharge();
  amplitudes_.resize(cluster.width(), cluster.avgCharge());
  filter_ = cluster.filter();

  float halfwidth_ = 0.5f * float(cluster.width());

  //initialize firstStrip_
  firstStrip_ = std::max(barycenter_ - halfwidth_, 0.f);

  if UNLIKELY (firstStrip_ + cluster.width() > maxStrips) {
    firstStrip_ = maxStrips - cluster.width();
  }
  firstStrip_ |= approximateMask;
}
