#ifndef RecoBTag_FeatureTools_SeedingTrackInfoBuilder_h
#define RecoBTag_FeatureTools_SeedingTrackInfoBuilder_h

#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "RecoBTag/TrackProbability/interface/HistogramProbabilityEstimator.h"

namespace btagbtvdeep {

  class SeedingTrackInfoBuilder {
  public:
    SeedingTrackInfoBuilder();

    void buildSeedingTrackInfo(const reco::TransientTrack* it,
                               const reco::Vertex& pv,
                               const reco::Jet& jet, /*GlobalVector jetdirection,*/
                               float mass,
                               const std::pair<bool, Measurement1D>& ip,
                               const std::pair<bool, Measurement1D>& ip2d,
                               float jet_distance,
                               float jaxis_dlength,
                               HistogramProbabilityEstimator* m_probabilityEstimator,
                               bool m_computeProbabilities);

    const float pt() const { return pt_; }
    const float eta() const { return eta_; }
    const float phi() const { return phi_; }
    const float mass() const { return mass_; }
    const float dz() const { return dz_; }
    const float dxy() const { return dxy_; }
    const float ip3d() const { return ip3D_; }
    const float sip3d() const { return sip3D_; }
    const float ip2d() const { return ip2D_; }
    const float sip2d() const { return sip2D_; }
    const float ip3d_Signed() const { return ip3D_signed_; }
    const float sip3d_Signed() const { return sip3D_signed_; }
    const float ip2d_Signed() const { return ip2D_signed_; }
    const float sip2d_Signed() const { return sip2D_signed_; }
    const float chi2reduced() const { return chi2reduced_; }
    const float nPixelHits() const { return nPixelHits_; }
    const float nHits() const { return nHits_; }
    const float jetAxisDistance() const { return jetAxisDistance_; }
    const float jetAxisDlength() const { return jetAxisDlength_; }
    const float trackProbability3D() const { return trackProbability3D_; }
    const float trackProbability2D() const { return trackProbability2D_; }

  private:
    float pt_;
    float eta_;
    float phi_;
    float mass_;
    float dz_;
    float dxy_;
    float ip3D_;
    float sip3D_;
    float ip2D_;
    float sip2D_;
    float ip3D_signed_;
    float sip3D_signed_;
    float ip2D_signed_;
    float sip2D_signed_;
    float chi2reduced_;
    float nPixelHits_;
    float nHits_;
    float jetAxisDistance_;
    float jetAxisDlength_;
    float trackProbability3D_;
    float trackProbability2D_;
  };
}  // namespace btagbtvdeep

#endif  //RecoBTag_FeatureTools_SeedingTrackInfoBuilder_h
