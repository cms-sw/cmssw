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

  explicit SiStripApproximateCluster(float barycenter, uint8_t width, float avgCharge) {
    barycenter_ = barycenter;
    width_ = width;
    avgCharge_ = avgCharge;
  }

  explicit SiStripApproximateCluster(const SiStripCluster& cluster);

  float barycenter() const { return barycenter_; }
  uint8_t width() const { return width_; }
  float avgCharge() const { return avgCharge_; }

private:
  float barycenter_ = 0;
  uint8_t width_ = 0;
  float avgCharge_ = 0;
};
#endif  // DATAFORMATS_SiStripApproximateCluster_H
