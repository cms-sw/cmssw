#ifndef DataFormats_SiStripCluster_SiStripApproximateCluster_h
#define DataFormats_SiStripCluster_SiStripApproximateCluster_h

#include "FWCore/Utilities/interface/typedefs.h"

class SiStripCluster;
class SiStripApproximateCluster {
public:
  SiStripApproximateCluster() {}

  explicit SiStripApproximateCluster(cms_uint16_t barycenter,
                                     cms_uint8_t width,
                                     cms_uint8_t avgCharge,
                                     bool isSaturated) {
    barycenter_ = barycenter;
    width_ = width;
    avgCharge_ = avgCharge;
    isSaturated_ = isSaturated;
  }

  explicit SiStripApproximateCluster(const SiStripCluster& cluster, unsigned int maxNSat);

  cms_uint16_t barycenter() const { return barycenter_; }
  cms_uint8_t width() const { return width_; }
  cms_uint8_t avgCharge() const { return avgCharge_; }
  bool isSaturated() const { return isSaturated_; }

private:
  cms_uint16_t barycenter_ = 0;
  cms_uint8_t width_ = 0;
  cms_uint8_t avgCharge_ = 0;
  bool isSaturated_ = false;
};
#endif  // DataFormats_SiStripCluster_SiStripApproximateCluster_h
