#ifndef DataFormats_SiStripCluster_SiStripApproximateCluster_h
#define DataFormats_SiStripCluster_SiStripApproximateCluster_h
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/typedefs.h"
#include <climits>
#include <bits/stdc++.h>

class SiStripCluster;
class SiStripApproximateCluster {
public:
  SiStripApproximateCluster() {}

  explicit SiStripApproximateCluster(cms_uint16_t barycenter,
                                     cms_uint8_t width,
                                     cms_uint8_t avgCharge,
                                     bool filter,
                                     bool isSaturated,
                                     bool peakFilter = false,
                                     cms_uint8_t version_ = 1)
      : barycenter_(barycenter),
        width_(width),
        avgCharge_(avgCharge),
        filter_(filter),
        isSaturated_(isSaturated),
        peakFilter_(peakFilter),
        version_(version_) {}

  explicit SiStripApproximateCluster(const SiStripCluster& cluster,
                                     unsigned int maxNSat,
                                     float hitPredPos,
                                     bool peakFilter,
                                     cms_uint8_t version_ = 1,
                                     float previous_barycenter = 0,
                                     unsigned int offset_module_change = 0);

  // getBarycenter returns the barycenter as a *float* in strips (e.g. 1.0 means center of strip 1)
  float getBarycenter(float previous_barycenter = 0, unsigned int offset_module_change = 0) const {
    switch (version_) {
      case 1:
        return barycenter_ * 0.1;  // in the old format barycenter_ is in tenths of strips
      case 2: {
        // Drop the first bit (encoding the saturation info)
        // barycenterRangeMax_ is 1 in 15 bits (the 16th bit is used to encode isSaturated_ info)
        double barycenter_decoded = (barycenter_ & barycenterRangeMax_);
        // rescale to get the original barycenter value (float)
        return barycenter_decoded / barycenterScale_ - offset_module_change + previous_barycenter;
      }
      default:
        throw cms::Exception("VersionNotSupported")
            << "Version " << int(version_) << " of SiStripApproximateCluster not supported";
    }
  }
  //   // kept for compatibility with v1, should be removed in the future
  //   cms_uint16_t barycenter(float previous_barycenter, unsigned int offset_module_change) const {
  //   switch (version_){
  //     case 2: {
  //       return std::round(getBarycenter(previous_barycenter, offset_module_change)*10.); // return barycenter in tenths of strips for compatibility with v1
  //     }
  //     default: throw cms::Exception("VersionNotSupported") << "Version " << int(version_) << " of SiStripApproximateCluster not supported for SiStripApproximateCluster::barycenter(float,unsigned int)";
  //   }
  //  }

  // getAvgCharge returns the average charge as a *float* in ADC counts (e.g. 1.0 means 1 ADC count)
  float getAvgCharge() const {
    switch (version_) {
      case 1:
        return avgCharge_;
      case 2: {
        // Drop the first two bits (encoding filter_ and peakFilter_ info)
        // avgChargeRangeMax_ is 1 in 6 bits (the 2 highest bits are used to encode filter_ and peakFilter_ info)
        double avgCharge_decoded = (avgCharge_ & avgChargeRangeMax_);
        // Rescale to get the original average charge (float)
        return (avgCharge_decoded)*avgChargeScale_ + avgChargeOffset_;
      }
      default:
        throw cms::Exception("VersionNotSupported")
            << "Version " << int(version_) << " of SiStripApproximateCluster not supported";
    }
  }

  //barycenter() returns barycenter position in tenths of strip (i.e. 10 means center of strip 1) (0-1536)
  //avgCharge() returns the average charge in ADC counts (0-255)
  //width() returns the cluster width (0-255)
  //version() returns true if the cluster is in the new format (Fall 2025)

  cms_uint16_t barycenter() const { return barycenter_; }
  cms_uint8_t width() const { return width_; }
  cms_uint8_t avgCharge() const {  // should be a float in the future instead of an int
    switch (version_) {
      case 1:
        return avgCharge_;
      default:
        throw cms::Exception("VersionNotSupported")
            << "Version " << int(version_)
            << " of SiStripApproximateCluster not supported for SiStripApproximateCluster::avgCharge()";
    }
  }

  bool filter() const {
    switch (version_) {
      case 1:
        return filter_;
      // In v2, filter_ info are encoded in avgCharge_
      case 2:
        return (avgCharge_ & (1 << kFilterBit));
      default:
        throw cms::Exception("VersionNotSupported")
            << "Version " << int(version_) << " of SiStripApproximateCluster not supported";
    }
  }
  bool peakFilter() const {
    switch (version_) {
      case 1:
        return peakFilter_;
      // In v2, filter_ and peakFilter_ info are encoded in avgCharge_
      case 2:
        return (avgCharge_ & (1 << kPeakFilterBit));
      default:
        throw cms::Exception("VersionNotSupported")
            << "Version " << int(version_) << " of SiStripApproximateCluster not supported";
    }
  }

  bool isSaturated() const {
    switch (version_) {
      case 1:
        return isSaturated_;
      // In v2, isSaturated_ info is encoded in barycenter_
      case 2:
        return (barycenter_ & (1 << kSaturatedBit));
      default:
        throw cms::Exception("VersionNotSupported")
            << "Version " << int(version_) << " of SiStripApproximateCluster not supported";
    }
  }
  char version() const { return version_; }

private:
  cms_uint16_t barycenter_ = 0;
  cms_uint8_t width_ = 0;
  cms_uint8_t avgCharge_ = 0;
  bool filter_ = false;
  bool isSaturated_ = false;
  bool peakFilter_ = false;
  // v2 --> new version
  cms_uint8_t version_ = 1;
  static constexpr double trimMaxADC_ = 30.;
  static constexpr double trimMaxFracTotal_ = .15;
  static constexpr double trimMaxFracNeigh_ = .25;
  static constexpr double maxTrimmedSizeDiffNeg_ = .7;
  static constexpr double maxTrimmedSizeDiffPos_ = 1.;

  ////// Encoding constants for v2 ///////////
  // maximum value of barycenter_ is 768 strips (128 strips/APV * 6 APVs)
  // multiplied by a factor 2 as we save the distance from the previous cluster, which can be in another module
  static constexpr double barycenterMax_ = 768. * 2;
  // get the total number of bits in barycenter_ (16 bits for cms_uint16_t)
  static constexpr int nbits_barycenter_ = sizeof(barycenter_) * CHAR_BIT;
  // position of the bit used to encode isSaturated_ in barycenter_
  static constexpr int kSaturatedBit = nbits_barycenter_ - 1;
  // get the largest number storable in barycenter_ with the remaining bits (2^15 -1 = 32767)
  static constexpr int barycenterRangeMax_ = (1 << (nbits_barycenter_ - 1)) - 1;

  // maximum value of avgCharge_ is 255 ADC counts
  static constexpr double avgChargeMax_ = 255.;
  // get the number of bits in avgCharge_ (8 bits for cms_uint8_t)
  static constexpr int nbits_avgCharge_ = sizeof(avgCharge_) * CHAR_BIT;
  // positions of the bit used to encode filter_ and peakFilter_ in avgCharge_
  static constexpr int kPeakFilterBit = nbits_avgCharge_ - 1;
  static constexpr int kFilterBit = nbits_avgCharge_ - 2;
  // get the largest number storable in avgCharge_ with the remaining bits (2^6 -1 = 63)
  static constexpr int avgChargeRangeMax_ = (1 << (nbits_avgCharge_ - 2)) - 1;

public:
  // constants used in compression and decompression,
  // floor approximate to integer to get no approximation error for integers
  static constexpr int barycenterScale_ = barycenterRangeMax_ / barycenterMax_;
  static constexpr int avgChargeScale_ = avgChargeMax_ / avgChargeRangeMax_;

  static constexpr float barycenterOffset_ = +0.5;  // to get no approximation error for integers
  static constexpr float avgChargeOffset_ = +2.0;   // to minimize bias
  ////////////////////////////////////////
};
#endif  // DataFormats_SiStripCluster_SiStripApproximateCluster_h
