#ifndef SiPixelTrackProbQXY_H
#define SiPixelTrackProbQXY_H

#include <cstdint>
#include <vector>

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  // Class defining the combined charge and shape probabilities for tracks that go through the SiPixel detector
  class SiPixelTrackProbQXY {
  public:
    SiPixelTrackProbQXY() {}

    SiPixelTrackProbQXY(float probQonTrack, float probXYonTrack, float probQonTrackNoL1, float probXYonTrackNoL1)
        : probQonTrack_(probQonTrack),
          probXYonTrack_(probXYonTrack),
          probQonTrackNoL1_(probQonTrackNoL1),
          probXYonTrackNoL1_(probXYonTrackNoL1) {}

    // Return the combined charge probabilities for tracks that go through the SiPixel detector
    float probQonTrack() const { return probQonTrack_; }

    // Return the combined shape probabilities for tracks that go through the SiPixel detector
    float probXYonTrack() const { return probXYonTrack_; }

    // Return the combined charge probabilities for tracks that go through the SiPixel detector
    // This version now excludes layer 1 which was known to be noisy for 2017/2018
    float probQonTrackNoL1() const { return probQonTrackNoL1_; }

    // Return the combined shape probabilities for tracks that go through the SiPixel detector
    // This version now excludes layer 1 which was known to be noisy for 2017/2018
    float probXYonTrackNoL1() const { return probXYonTrackNoL1_; }

  private:
    float probQonTrack_;
    float probXYonTrack_;
    float probQonTrackNoL1_;
    float probXYonTrackNoL1_;
  };

  typedef std::vector<SiPixelTrackProbQXY> SiPixelTrackProbQXYCollection;
  typedef edm::Ref<SiPixelTrackProbQXYCollection> SiPixelTrackProbQXYRef;
  typedef edm::RefProd<SiPixelTrackProbQXYCollection> SiPixelTrackProbQXYRefProd;
  typedef edm::RefVector<SiPixelTrackProbQXYCollection> SiPixelTrackProbQXYRefVector;
  typedef edm::Association<SiPixelTrackProbQXYCollection> SiPixelTrackProbQXYAssociation;

}  // namespace reco
#endif
