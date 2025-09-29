#include "DataFormats/SiStripCluster/interface/SiStripApproximateCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include <algorithm>
#include <cassert>
#include <cmath>

SiStripApproximateCluster::SiStripApproximateCluster(const SiStripCluster& cluster,
                                                     unsigned int maxNSat,
                                                     float hitPredPos,
                                                     bool peakFilter,
                                                     unsigned char version,
                                                     float previous_barycenter,
                                                     unsigned int offset_module_change) {
  barycenter_ = std::round(cluster.barycenter() * 10);
  width_ = cluster.size();
  avgCharge_ = cluster.charge() / cluster.size();
  filter_ = false;
  isSaturated_ = false;
  peakFilter_ = peakFilter;
  version_ = version;

  //mimicing the algorithm used in StripSubClusterShapeTrajectoryFilter...
  //Looks for 3 adjacent saturated strips (ADC>=254)
  const auto& ampls = cluster.amplitudes();
  unsigned int thisSat = (ampls[0] >= 254), maxSat = thisSat;
  for (unsigned int i = 1, n = ampls.size(); i < n; ++i) {
    if (ampls[i] >= 254) {
      thisSat++;
    } else if (thisSat > 0) {
      maxSat = std::max<int>(maxSat, thisSat);
      thisSat = 0;
    }
  }
  if (thisSat > 0) {
    maxSat = std::max<int>(maxSat, thisSat);
  }
  if (maxSat >= maxNSat) {
    filter_ = true;
    isSaturated_ = true;
  }

  unsigned int hitStripsTrim = ampls.size();
  int sum = std::accumulate(ampls.begin(), ampls.end(), 0);
  uint8_t trimCut = std::min<uint8_t>(trimMaxADC_, std::floor(trimMaxFracTotal_ * sum));
  auto begin = ampls.begin();
  auto last = ampls.end() - 1;
  while (hitStripsTrim > 1 && (*begin < std::max<uint8_t>(trimCut, trimMaxFracNeigh_ * (*(begin + 1))))) {
    hitStripsTrim--;
    ++begin;
  }
  while (hitStripsTrim > 1 && (*last < std::max<uint8_t>(trimCut, trimMaxFracNeigh_ * (*(last - 1))))) {
    hitStripsTrim--;
    --last;
  }
  if (hitStripsTrim < std::floor(std::abs(hitPredPos) - maxTrimmedSizeDiffNeg_)) {
    filter_ = false;
  } else if (hitStripsTrim <= std::ceil(std::abs(hitPredPos) + maxTrimmedSizeDiffPos_)) {
    filter_ = true;
  } else {
    filter_ = peakFilter_;
  }

  if (version_ == 2) {
    // Compression of avgCharge_ to integer
    avgCharge_ = floor((float(cluster.charge()) / cluster.size()) / avgChargeScale_);
    // In v2, we encode the filter_ and peakFilter_ info in avgCharge_ as the two highest bits
    assert(avgCharge_ <= avgChargeRangeMax_ && "Setting avgCharge > avgChargeRangeMax_");
    // filter_ and peakFilter_ are encoded in the two highest bits of avgCharge_
    avgCharge_ = (avgCharge_ | (filter_ << kFilterBit));
    assert(avgCharge_ <= ((1 << (nbits_avgCharge_ - 1)) - 1) && "Setting avgCharge with filter > max single bit range");
    avgCharge_ = (avgCharge_ | (peakFilter_ << kPeakFilterBit));
    assert(avgCharge_ <= ((1 << nbits_avgCharge_) - 1) && "Setting avgCharge with peakFilter > max full range");

    // Compression of barycenter_ to integer [note: it contains the distance from the previous cluster]
    barycenter_ = round(float(cluster.barycenter() - previous_barycenter + (offset_module_change)) * barycenterScale_);
    assert(barycenter_ <= barycenterRangeMax_ && "Setting barycenter > barycenterRangeMax_");
    // isSaturated_ is encoded in the highest bit of barycenter_
    barycenter_ = (barycenter_ | (isSaturated_ << kSaturatedBit));
    assert(barycenter_ <= ((1 << nbits_barycenter_) - 1) && "Setting barycenter with isSaturated > max full range");

    // Flags set to false to reduce event size (they should be removed when moving to v2 only)
    filter_ = false;
    isSaturated_ = false;
    peakFilter_ = false;
  }
}
