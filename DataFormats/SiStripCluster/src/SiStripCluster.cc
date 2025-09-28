#include "FWCore/Utilities/interface/Likely.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "FWCore/Utilities/interface/Exception.h"

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

SiStripCluster::SiStripCluster(const SiStripApproximateCluster cluster,
                               const uint16_t maxStrips,
                               float previous_barycenter,
                               unsigned int offset_module_change)
    : error_x(-99999.9) {
  switch (cluster.version()) {
    case 1: {
      barycenter_ = cluster.barycenter() / 10.0;
      charge_ = cluster.width() * cluster.avgCharge();
      amplitudes_.resize(cluster.width(), cluster.avgCharge());
      break;
    }
    case 2: {
      barycenter_ = cluster.getBarycenter(previous_barycenter, offset_module_change);
      charge_ = round(cluster.getAvgCharge() * cluster.width());
      int amplitude = ceil(cluster.getAvgCharge());
      amplitudes_.resize(cluster.width(), amplitude);
      int excessToBeFixed = amplitude * cluster.width() - charge_;
      // Reduce the first and last amplitude to maintain the same average charge but reduce the width by 1
      if (cluster.width() > 1 && excessToBeFixed != 0) {
        if (excessToBeFixed % 2 == 0) {
          amplitudes_.front() = amplitude - excessToBeFixed / 2;
          amplitudes_.back() = amplitude - excessToBeFixed / 2;
          // assert(amplitude - excessToBeFixed / 2 >=0);
        } else {
          // if the excess is odd, we cannot divide it equally between the first and last strip
          // we add one unit to the front and subtract one unit to the back to keep the total charge unchanged
          if (int(cluster.getAvgCharge()) % 2 == 0) {  // avoid bias with rounding
            amplitudes_.front() = amplitude - (excessToBeFixed / 2 + 1);
            amplitudes_.back() = amplitude - (excessToBeFixed / 2);
            // assert(amplitude - (excessToBeFixed / 2 + 1) >=0);
          } else {
            amplitudes_.back() = amplitude - (excessToBeFixed / 2 + 1);
            amplitudes_.front() = amplitude - (excessToBeFixed / 2);
            // assert(amplitude - (excessToBeFixed / 2 + 1) >=0);
          }
        }
      }
      // assert( int(std::accumulate(amplitudes_.begin(), amplitudes_.end(), 0.0f)) == charge_);
      break;
    }
    default:
      throw cms::Exception("VersionNotSupported")
          << "SiStripCluster.cc. Version " << int(cluster.version()) << " of SiStripApproximateCluster not supported";
  }
  filter_ = cluster.filter();

  float halfwidth_ = 0.5f * float(cluster.width());

  //initialize firstStrip_
  firstStrip_ = std::max(barycenter_ - halfwidth_, 0.f);

  if UNLIKELY (firstStrip_ + cluster.width() > maxStrips) {
    firstStrip_ = maxStrips - cluster.width();
  }
  firstStrip_ |= approximateMask;
}
