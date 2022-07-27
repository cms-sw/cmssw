#ifndef DATAFORMATS_SISTRIPAPPROXIMATECLUSTER_H
#define DATAFORMATS_SISTRIPAPPROXIMATECLUSTER_H

#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

class SiStripCluster;
class SiStripApproximateCluster {
public:
  SiStripApproximateCluster() {}

  explicit SiStripApproximateCluster(float barycenter, uint8_t width, float avgCharge, bool isSaturated) {
    barycenter_ = barycenter;
    width_ = width;
    avgCharge_ = avgCharge;
    isSaturated_ = isSaturated;
  }

  explicit SiStripApproximateCluster(const SiStripCluster& cluster, unsigned int maxNSat);

  float barycenter() const { return barycenter_; }
  uint8_t width() const { return width_; }
  float avgCharge() const { return avgCharge_; }
  bool isSaturated() const { return isSaturated_; }

private:
  float barycenter_ = 0;
  uint8_t width_ = 0;
  float avgCharge_ = 0;
  bool isSaturated_ = false;
};
#endif  // DATAFORMATS_SiStripApproximateCluster_H
