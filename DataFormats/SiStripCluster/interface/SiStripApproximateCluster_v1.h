#ifndef DataFormats_SiStripCluster_SiStripApproximateCluster_v1_h
#define DataFormats_SiStripCluster_SiStripApproximateCluster_v1_h

#include "FWCore/Utilities/interface/typedefs.h"
#include "assert.h"

class SiStripCluster;

namespace v1 {
  class SiStripApproximateCluster {
  public:
    SiStripApproximateCluster() {}

    explicit SiStripApproximateCluster(cms_uint16_t compBarycenter, cms_uint8_t width, cms_uint8_t compavgCharge)
        : compBarycenter_(compBarycenter), width_(width), compavgCharge_(compavgCharge) {}

    explicit SiStripApproximateCluster(const SiStripCluster& cluster,
                                       unsigned int maxNSat,
                                       float hitPredPos,
                                       float& previous_cluster,
                                       unsigned int& module_length,
                                       unsigned int& previous_module_length,
                                       bool peakFilter);

    const cms_uint16_t compBarycenter() const { return compBarycenter_; }

    float barycenter(float previous_barycenter = 0,
                     unsigned int module_length = 0,
                     unsigned int previous_module_length = 0) const {
      float barycenter;
      cms_uint16_t compBarycenter = (compBarycenter_ & 0x7FFF);
      if (previous_barycenter == -999)
        barycenter = compBarycenter * maxBarycenter_ / maxRange_;
      else {
        barycenter = ((compBarycenter * maxBarycenter_ / maxRange_) - (module_length - previous_module_length)) +
                     previous_barycenter;
      }
      assert(barycenter <= maxBarycenter_ && "Returning barycenter > maxBarycenter");
      return barycenter;
    }
    cms_uint8_t width() const { return width_; }
    float avgCharge() const {
      cms_uint8_t compavgCharge = (compavgCharge_ & 0x3F);
      float avgCharge_ = compavgCharge * maxavgCharge_ / maxavgChargeRange_;
      assert(avgCharge_ <= maxavgCharge_ && "Returning avgCharge > maxavgCharge");
      return avgCharge_;
    }
    bool filter() const { return (compavgCharge_ & (1 << kfilterMask)); }
    bool isSaturated() const { return (compavgCharge_ & (1 << kSaturatedMask)); }
    bool peakFilter() const { return (compBarycenter_ & (1 << kpeakFilterMask)); }

  private:
    cms_uint16_t compBarycenter_ = 0;
    cms_uint8_t width_ = 0;
    cms_uint8_t compavgCharge_ = 0;
    static constexpr double maxRange_ = 32767;
    static constexpr double maxBarycenter_ = 1536.;
    static constexpr double maxavgChargeRange_ = 63;
    static constexpr double maxavgCharge_ = 255.;
    static constexpr double trimMaxADC_ = 30.;
    static constexpr double trimMaxFracTotal_ = .15;
    static constexpr double trimMaxFracNeigh_ = .25;
    static constexpr double maxTrimmedSizeDiffNeg_ = .7;
    static constexpr double maxTrimmedSizeDiffPos_ = 1.;
    static constexpr int kfilterMask = 6;
    static constexpr int kpeakFilterMask = 7;
    static constexpr int kSaturatedMask = 15;
  };
}  // namespace v1
#endif  // DataFormats_SiStripCluster_SiStripApproximateCluster_h
