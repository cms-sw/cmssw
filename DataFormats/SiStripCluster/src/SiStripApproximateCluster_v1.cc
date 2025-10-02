#include "DataFormats/SiStripCluster/interface/SiStripApproximateCluster_v1.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include <algorithm>
#include <cassert>
#include <cmath>
v1::SiStripApproximateCluster::SiStripApproximateCluster(const SiStripCluster& cluster,
                                                         unsigned int maxNSat,
                                                         float hitPredPos,
                                                         float& previous_cluster,
                                                         unsigned int& module_length,
                                                         unsigned int& previous_module_length,
                                                         bool peakFilter) {
  bool filter_, isSaturated_, peakFilter_;
  if (previous_cluster == -999.)
    compBarycenter_ = std::round(cluster.barycenter() * maxRange_ / maxBarycenter_);
  else
    compBarycenter_ =
        std::round(((cluster.barycenter() - previous_cluster) + (module_length - previous_module_length)) * maxRange_ /
                   maxBarycenter_);
  previous_cluster = barycenter(previous_cluster, module_length, previous_module_length);
  assert(cluster.barycenter() <= maxBarycenter_ && "Got a barycenter > maxBarycenter");
  assert(compBarycenter_ <= maxRange_ && "Filling compBarycenter > maxRange");
  width_ = std::min(255, (int)cluster.size());  //cluster.size();
  float avgCharge_ = cluster.charge() * 1. / width_;
  assert(avgCharge_ <= maxavgCharge_ && "Got a avgCharge > maxavgCharge");
  compavgCharge_ = std::round(avgCharge_ * maxavgChargeRange_ / maxavgCharge_);
  assert(compavgCharge_ <= maxavgChargeRange_ && "Filling compavgCharge > maxavgChargeRange");
  filter_ = false;
  isSaturated_ = false;
  peakFilter_ = peakFilter;

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
  compavgCharge_ = (compavgCharge_ | (filter_ << kfilterMask));
  compavgCharge_ = (compavgCharge_ | (peakFilter_ << kpeakFilterMask));
  compBarycenter_ = (compBarycenter_ | (isSaturated_ << kSaturatedMask));
}
