#ifndef RecoMTD_TimingTools_TrackSegments_h
#define RecoMTD_TimingTools_TrackSegments_h

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <utility>
#include <vector>

#include <CLHEP/Units/GlobalPhysicalConstants.h>

#include "DataFormats/Math/interface/GeantUnits.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace mtd {

  constexpr float c_cm_ns = geant_units::operators::convertMmToCm(CLHEP::c_light);  // [mm/ns] -> [cm/ns]
  constexpr float c_inv = 1.0f / c_cm_ns;

  class TrackSegments {
  public:
    TrackSegments() {
      sigmaTofs_.reserve(30);  // observed upper limit on nSegments
    }

    inline uint32_t addSegment(float tPath, float tMom2, float sigmaMom) {
      segmentPathOvc_.emplace_back(tPath * c_inv);
      segmentMom2_.emplace_back(tMom2);
      segmentSigmaMom_.emplace_back(sigmaMom);
      nSegment_ += 1;

      LogTrace("TrackExtenderWithMTD") << "addSegment # " << nSegment_ << " s = " << tPath
                                       << " p = " << std::sqrt(tMom2) << " sigma_p = " << sigmaMom
                                       << " sigma_p/p = " << sigmaMom / std::sqrt(tMom2) * 100 << " %";

      return nSegment_;
    }

    inline float computeTof(float mass_inv2) const {
      float tof(0.f);
      for (uint32_t iSeg = 0; iSeg < nSegment_; iSeg++) {
        float gammasq = 1.f + segmentMom2_[iSeg] * mass_inv2;
        float beta = std::sqrt(1.f - 1.f / gammasq);
        tof += segmentPathOvc_[iSeg] / beta;

        LogTrace("TrackExtenderWithMTD") << " TOF Segment # " << iSeg + 1 << " p = " << std::sqrt(segmentMom2_[iSeg])
                                         << " tof = " << tof;

#ifdef EDM_ML_DEBUG
        float sigma_tof = segmentPathOvc_[iSeg] * segmentSigmaMom_[iSeg] /
                          (segmentMom2_[iSeg] * sqrt(segmentMom2_[iSeg] + 1 / mass_inv2) * mass_inv2);

        LogTrace("TrackExtenderWithMTD") << "TOF Segment # " << iSeg + 1 << std::fixed << std::setw(6)
                                         << " tof segment = " << segmentPathOvc_[iSeg] / beta << std::scientific
                                         << "+/- " << sigma_tof << std::fixed
                                         << "(rel. err. = " << sigma_tof / (segmentPathOvc_[iSeg] / beta) * 100
                                         << " %)";
#endif
      }

      return tof;
    }

    inline float computeSigmaTof(float mass_inv2) {
      float sigmatof = 0.;

      // remove previously calculated sigmaTofs
      sigmaTofs_.clear();

      // compute sigma(tof) on each segment first by propagating sigma(p)
      // also add diagonal terms to sigmatof
      float sigma = 0.;
      for (uint32_t iSeg = 0; iSeg < nSegment_; iSeg++) {
        sigma = segmentPathOvc_[iSeg] * segmentSigmaMom_[iSeg] /
                (segmentMom2_[iSeg] * sqrt(segmentMom2_[iSeg] + 1 / mass_inv2) * mass_inv2);
        sigmaTofs_.push_back(sigma);

        sigmatof += sigma * sigma;
      }

      // compute sigma on sum of tofs assuming full correlation between segments
      for (uint32_t iSeg = 0; iSeg < nSegment_; iSeg++) {
        for (uint32_t jSeg = iSeg + 1; jSeg < nSegment_; jSeg++) {
          sigmatof += 2 * sigmaTofs_[iSeg] * sigmaTofs_[jSeg];
        }
      }

      return sqrt(sigmatof);
    }

    inline uint32_t size() const { return nSegment_; }

    inline uint32_t removeFirstSegment() {
      if (nSegment_ > 0) {
        segmentPathOvc_.erase(segmentPathOvc_.begin());
        segmentMom2_.erase(segmentMom2_.begin());
        nSegment_--;
      }
      return nSegment_;
    }

    inline std::pair<float, float> getSegmentPathAndMom2(uint32_t iSegment) const {
      if (iSegment >= nSegment_) {
        throw cms::Exception("TrackExtenderWithMTD") << "Requesting non existing track segment #" << iSegment;
      }
      return std::make_pair(segmentPathOvc_[iSegment], segmentMom2_[iSegment]);
    }

    uint32_t nSegment_ = 0;
    std::vector<float> segmentPathOvc_;
    std::vector<float> segmentMom2_;
    std::vector<float> segmentSigmaMom_;

    std::vector<float> sigmaTofs_;
  };

}  // namespace mtd

#endif
