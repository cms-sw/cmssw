#include "RecoBTag/FeatureTools/interface/TrackPairInfoBuilder.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

namespace btagbtvdeep {

  TrackPairInfoBuilder::TrackPairInfoBuilder()
      :

        track_pt_(0),
        track_eta_(0),
        track_phi_(0),
        track_dz_(0),
        track_dxy_(0),

        pca_distance_(0),
        pca_significance_(0),

        pcaSeed_x_(0),
        pcaSeed_y_(0),
        pcaSeed_z_(0),
        pcaSeed_xerr_(0),
        pcaSeed_yerr_(0),
        pcaSeed_zerr_(0),
        pcaTrack_x_(0),
        pcaTrack_y_(0),
        pcaTrack_z_(0),
        pcaTrack_xerr_(0),
        pcaTrack_yerr_(0),
        pcaTrack_zerr_(0),

        dotprodTrack_(0),
        dotprodSeed_(0),
        pcaSeed_dist_(0),
        pcaTrack_dist_(0),

        track_candMass_(0),
        track_ip2d_(0),
        track_ip2dSig_(0),
        track_ip3d_(0),
        track_ip3dSig_(0),

        dotprodTrackSeed2D_(0),
        dotprodTrackSeed2DV_(0),
        dotprodTrackSeed3D_(0),
        dotprodTrackSeed3DV_(0),

        pca_jetAxis_dist_(0),
        pca_jetAxis_dotprod_(0),
        pca_jetAxis_dEta_(0),
        pca_jetAxis_dPhi_(0)

  {}

  void TrackPairInfoBuilder::buildTrackPairInfo(const reco::TransientTrack* it,
                                                const reco::TransientTrack* tt,
                                                const reco::Vertex& pv,
                                                float mass,
                                                GlobalVector jetdirection,
                                                const std::pair<bool, Measurement1D>& t_ip,
                                                const std::pair<bool, Measurement1D>& t_ip2d) {
    GlobalPoint pvp(pv.x(), pv.y(), pv.z());

    VertexDistance3D distanceComputer;
    TwoTrackMinimumDistance dist;

    auto const& iImpactState = it->impactPointState();
    auto const& tImpactState = tt->impactPointState();

    if (dist.calculate(tImpactState, iImpactState)) {
      GlobalPoint ttPoint = dist.points().first;
      GlobalError ttPointErr = tImpactState.cartesianError().position();
      GlobalPoint seedPosition = dist.points().second;
      GlobalError seedPositionErr = iImpactState.cartesianError().position();

      Measurement1D m =
          distanceComputer.distance(VertexState(seedPosition, seedPositionErr), VertexState(ttPoint, ttPointErr));

      GlobalPoint cp(dist.crossingPoint());

      GlobalVector pairMomentum((Basic3DVector<float>)(it->track().momentum() + tt->track().momentum()));
      GlobalVector pvToPCA(cp - pvp);

      float pvToPCAseed = (seedPosition - pvp).mag();
      float pvToPCAtrack = (ttPoint - pvp).mag();
      float distance = dist.distance();

      GlobalVector trackDir2D(tImpactState.globalDirection().x(), tImpactState.globalDirection().y(), 0.);
      GlobalVector seedDir2D(iImpactState.globalDirection().x(), iImpactState.globalDirection().y(), 0.);
      GlobalVector trackPCADir2D(ttPoint.x() - pvp.x(), ttPoint.y() - pvp.y(), 0.);
      GlobalVector seedPCADir2D(seedPosition.x() - pvp.x(), seedPosition.y() - pvp.y(), 0.);

      float dotprodTrack = (ttPoint - pvp).unit().dot(tImpactState.globalDirection().unit());
      float dotprodSeed = (seedPosition - pvp).unit().dot(iImpactState.globalDirection().unit());

      Line::PositionType pos(pvp);
      Line::DirectionType dir(jetdirection);
      Line::DirectionType pairMomentumDir(pairMomentum);
      Line jetLine(pos, dir);
      Line PCAMomentumLine(cp, pairMomentumDir);

      track_pt_ = tt->track().pt();
      track_eta_ = tt->track().eta();
      track_phi_ = tt->track().phi();
      track_dz_ = tt->track().dz(pv.position());
      track_dxy_ = tt->track().dxy(pv.position());

      pca_distance_ = distance;
      pca_significance_ = m.significance();

      pcaSeed_x_ = seedPosition.x();
      pcaSeed_y_ = seedPosition.y();
      pcaSeed_z_ = seedPosition.z();
      pcaSeed_xerr_ = seedPositionErr.cxx();
      pcaSeed_yerr_ = seedPositionErr.cyy();
      pcaSeed_zerr_ = seedPositionErr.czz();
      pcaTrack_x_ = ttPoint.x();
      pcaTrack_y_ = ttPoint.y();
      pcaTrack_z_ = ttPoint.z();
      pcaTrack_xerr_ = ttPointErr.cxx();
      pcaTrack_yerr_ = ttPointErr.cyy();
      pcaTrack_zerr_ = ttPointErr.czz();

      dotprodTrack_ = dotprodTrack;
      dotprodSeed_ = dotprodSeed;
      pcaSeed_dist_ = pvToPCAseed;
      pcaTrack_dist_ = pvToPCAtrack;

      track_candMass_ = mass;
      track_ip2d_ = t_ip2d.second.value();
      track_ip2dSig_ = t_ip2d.second.significance();
      track_ip3d_ = t_ip.second.value();
      track_ip3dSig_ = t_ip.second.significance();

      dotprodTrackSeed2D_ = trackDir2D.unit().dot(seedDir2D.unit());
      dotprodTrackSeed3D_ = iImpactState.globalDirection().unit().dot(tImpactState.globalDirection().unit());
      dotprodTrackSeed2DV_ = trackPCADir2D.unit().dot(seedPCADir2D.unit());
      dotprodTrackSeed3DV_ = (seedPosition - pvp).unit().dot((ttPoint - pvp).unit());

      pca_jetAxis_dist_ = jetLine.distance(cp).mag();
      pca_jetAxis_dotprod_ = pairMomentum.unit().dot(jetdirection.unit());
      pca_jetAxis_dEta_ = std::fabs(pvToPCA.eta() - jetdirection.eta());
      pca_jetAxis_dPhi_ = std::fabs(pvToPCA.phi() - jetdirection.phi());
    }
  }

}  // namespace btagbtvdeep
