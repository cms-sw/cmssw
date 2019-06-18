#ifndef DataFormats_BTauReco_SeedingTrackFeatures_h
#define DataFormats_BTauReco_SeedingTrackFeatures_h

#include <vector>
#include "DataFormats/BTauReco/interface/TrackPairFeatures.h"

namespace btagbtvdeep {

  class SeedingTrackFeatures {
  public:
    float pt;
    float eta;
    float phi;
    float mass;
    float dz;
    float dxy;
    float ip3D;
    float sip3D;
    float ip2D;
    float sip2D;
    float signedIp3D;
    float signedSip3D;
    float signedIp2D;
    float signedSip2D;
    float trackProbability3D;
    float trackProbability2D;
    float chi2reduced;
    float nPixelHits;
    float nHits;
    float jetAxisDistance;
    float jetAxisDlength;

    std::vector<btagbtvdeep::TrackPairFeatures> nearTracks;
  };

}  // namespace btagbtvdeep

#endif
