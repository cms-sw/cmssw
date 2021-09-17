#ifndef RecoBTag_FeatureTools_TrackPairInfoBuilder_h
#define RecoBTag_FeatureTools_TrackPairInfoBuilder_h

#include "DataFormats/GeometrySurface/interface/Line.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"

namespace btagbtvdeep {

  class TrackPairInfoBuilder {
  public:
    TrackPairInfoBuilder();

    void buildTrackPairInfo(const reco::TransientTrack* it,
                            const reco::TransientTrack* tt,
                            const reco::Vertex& pv,
                            float mass,
                            GlobalVector jetdirection,
                            const std::pair<bool, Measurement1D>& t_ip,
                            const std::pair<bool, Measurement1D>& t_ip2d);

    const float track_pt() const { return track_pt_; }
    const float track_eta() const { return track_eta_; }
    const float track_phi() const { return track_phi_; }
    const float track_dz() const { return track_dz_; }
    const float track_dxy() const { return track_dxy_; }
    const float pca_distance() const { return pca_distance_; }
    const float pca_significance() const { return pca_significance_; }
    const float pcaSeed_x() const { return pcaSeed_x_; }
    const float pcaSeed_y() const { return pcaSeed_y_; }
    const float pcaSeed_z() const { return pcaSeed_z_; }
    const float pcaSeed_xerr() const { return pcaSeed_xerr_; }
    const float pcaSeed_yerr() const { return pcaSeed_yerr_; }
    const float pcaSeed_zerr() const { return pcaSeed_zerr_; }
    const float pcaTrack_x() const { return pcaTrack_x_; }
    const float pcaTrack_y() const { return pcaTrack_y_; }
    const float pcaTrack_z() const { return pcaTrack_z_; }
    const float pcaTrack_xerr() const { return pcaTrack_xerr_; }
    const float pcaTrack_yerr() const { return pcaTrack_yerr_; }
    const float pcaTrack_zerr() const { return pcaTrack_zerr_; }
    const float dotprodTrack() const { return dotprodTrack_; }
    const float dotprodSeed() const { return dotprodSeed_; }
    const float pcaSeed_dist() const { return pcaSeed_dist_; }
    const float pcaTrack_dist() const { return pcaTrack_dist_; }
    const float track_candMass() const { return track_candMass_; }
    const float track_ip2d() const { return track_ip2d_; }
    const float track_ip2dSig() const { return track_ip2dSig_; }
    const float track_ip3d() const { return track_ip3d_; }
    const float track_ip3dSig() const { return track_ip3dSig_; }
    const float dotprodTrackSeed2D() const { return dotprodTrackSeed2D_; }
    const float dotprodTrackSeed2DV() const { return dotprodTrackSeed2DV_; }
    const float dotprodTrackSeed3D() const { return dotprodTrackSeed3D_; }
    const float dotprodTrackSeed3DV() const { return dotprodTrackSeed3DV_; }
    const float pca_jetAxis_dist() const { return pca_jetAxis_dist_; }
    const float pca_jetAxis_dotprod() const { return pca_jetAxis_dotprod_; }
    const float pca_jetAxis_dEta() const { return pca_jetAxis_dEta_; }
    const float pca_jetAxis_dPhi() const { return pca_jetAxis_dPhi_; }

  private:
    float track_pt_;
    float track_eta_;
    float track_phi_;
    float track_dz_;
    float track_dxy_;
    float pca_distance_;
    float pca_significance_;
    float pcaSeed_x_;
    float pcaSeed_y_;
    float pcaSeed_z_;
    float pcaSeed_xerr_;
    float pcaSeed_yerr_;
    float pcaSeed_zerr_;
    float pcaTrack_x_;
    float pcaTrack_y_;
    float pcaTrack_z_;
    float pcaTrack_xerr_;
    float pcaTrack_yerr_;
    float pcaTrack_zerr_;
    float dotprodTrack_;
    float dotprodSeed_;
    float pcaSeed_dist_;
    float pcaTrack_dist_;
    float track_candMass_;
    float track_ip2d_;
    float track_ip2dSig_;
    float track_ip3d_;
    float track_ip3dSig_;
    float dotprodTrackSeed2D_;
    float dotprodTrackSeed2DV_;
    float dotprodTrackSeed3D_;
    float dotprodTrackSeed3DV_;
    float pca_jetAxis_dist_;
    float pca_jetAxis_dotprod_;
    float pca_jetAxis_dEta_;
    float pca_jetAxis_dPhi_;
  };

}  // namespace btagbtvdeep

#endif  //RecoBTag_FeatureTools_TrackPairInfoBuilder_h
