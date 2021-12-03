#ifndef DataFormats_TrackReco_SiPixelTrackProbQXY_H
#define DataFormats_TrackReco_SiPixelTrackProbQXY_H

#include <cstdint>
#include <vector>

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  // Class defining the combined charge and shape probabilities for tracks that go through the SiPixel detector
  class SiPixelTrackProbQXY {
  public:
    SiPixelTrackProbQXY() = default;

    SiPixelTrackProbQXY(float probQonTrack, float probXYonTrack, float probQonTrackNoLayer1, float probXYonTrackNoLayer1)
        : probQonTrack_(probQonTrack),
          probXYonTrack_(probXYonTrack),
          probQonTrackNoLayer1_(probQonTrackNoLayer1),
          probXYonTrackNoLayer1_(probXYonTrackNoLayer1) {}

    //! Return the combined charge probabilities for tracks that go through the SiPixel detector
    float probQ() const { return probQonTrack_; }

    //! Return the combined shape probabilities for tracks that go through the SiPixel detector
    float probXY() const { return probXYonTrack_; }

    //! Return the combined charge probabilities for tracks that go through the SiPixel detector
    //! This version now excludes layer 1 which was known to be noisy for 2017/2018
    float probQNoLayer1() const { return probQonTrackNoLayer1_; }

    //! Return the combined shape probabilities for tracks that go through the SiPixel detector
    //! This version now excludes layer 1 which was known to be noisy for 2017/2018
    float probXYNoLayer1() const { return probXYonTrackNoLayer1_; }

  private:
    float probQonTrack_ = 0;
    float probXYonTrack_ = 0;
    float probQonTrackNoLayer1_ = 0;
    float probXYonTrackNoLayer1_ = 0;
  };

  typedef std::vector<SiPixelTrackProbQXY> SiPixelTrackProbQXYCollection;
  typedef edm::Ref<SiPixelTrackProbQXYCollection> SiPixelTrackProbQXYRef;
  typedef edm::RefProd<SiPixelTrackProbQXYCollection> SiPixelTrackProbQXYRefProd;
  typedef edm::RefVector<SiPixelTrackProbQXYCollection> SiPixelTrackProbQXYRefVector;
  typedef edm::Association<SiPixelTrackProbQXYCollection> SiPixelTrackProbQXYAssociation;

}  // namespace reco
#endif
