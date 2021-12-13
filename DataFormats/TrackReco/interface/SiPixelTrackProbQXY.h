#ifndef DataFormats_TrackReco_SiPixelTrackProbQXY_H
#define DataFormats_TrackReco_SiPixelTrackProbQXY_H

#include <cstdint>
#include <vector>

#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/ValueMap.h"

namespace reco {
  // Class defining the combined charge and shape probabilities for tracks that go through the SiPixel detector
  class SiPixelTrackProbQXY {
  public:
    SiPixelTrackProbQXY() = default;

    SiPixelTrackProbQXY(float probQonTrack, float probXYonTrack)
        : probQonTrack_(probQonTrack), probXYonTrack_(probXYonTrack) {}

    //! Return the combined charge probabilities for tracks that go through the SiPixel detector
    float probQ() const { return probQonTrack_; }

    //! Return the combined shape probabilities for tracks that go through the SiPixel detector
    float probXY() const { return probXYonTrack_; }

  private:
    float probQonTrack_ = 0;
    float probXYonTrack_ = 0;
  };

  typedef std::vector<SiPixelTrackProbQXY> SiPixelTrackProbQXYCollection;
  typedef edm::ValueMap<SiPixelTrackProbQXY> SiPixelTrackProbQXYValueMap;
  typedef edm::Ref<SiPixelTrackProbQXYCollection> SiPixelTrackProbQXYRef;
  typedef edm::RefProd<SiPixelTrackProbQXYCollection> SiPixelTrackProbQXYRefProd;
  typedef edm::RefVector<SiPixelTrackProbQXYCollection> SiPixelTrackProbQXYRefVector;

}  // namespace reco
#endif
