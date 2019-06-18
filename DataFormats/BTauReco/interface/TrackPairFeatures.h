#ifndef DataFormats_BTauReco_TrackPairFeatures_h
#define DataFormats_BTauReco_TrackPairFeatures_h

namespace btagbtvdeep {

  class TrackPairFeatures {
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
    float distPCA;
    float dsigPCA;
    float x_PCAonSeed;
    float y_PCAonSeed;
    float z_PCAonSeed;
    float xerr_PCAonSeed;
    float yerr_PCAonSeed;
    float zerr_PCAonSeed;
    float x_PCAonTrack;
    float y_PCAonTrack;
    float z_PCAonTrack;
    float xerr_PCAonTrack;
    float yerr_PCAonTrack;
    float zerr_PCAonTrack;
    float dotprodTrack;
    float dotprodSeed;
    float dotprodTrackSeed2D;
    float dotprodTrackSeed3D;
    float dotprodTrackSeedVectors2D;
    float dotprodTrackSeedVectors3D;
    float pvd_PCAonSeed;
    float pvd_PCAonTrack;
    float dist_PCAjetAxis;
    float dotprod_PCAjetMomenta;
    float deta_PCAjetDirs;
    float dphi_PCAjetDirs;
  };

}  // namespace btagbtvdeep

#endif
