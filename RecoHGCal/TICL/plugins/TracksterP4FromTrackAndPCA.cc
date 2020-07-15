// Plugin for getting the four-vector of a Trackster from the track, if
// present, or from PCA decomposition.

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoHGCal/TICL/interface/TracksterMomentumPluginBase.h"

namespace ticl {
  class TracksterP4FromTrackAndPCA final : public TracksterMomentumPluginBase {
  public:
    explicit TracksterP4FromTrackAndPCA(const edm::ParameterSet&, edm::ConsumesCollector&& iC);
    void setP4(const std::vector<const Trackster*>& tracksters,
               std::vector<TICLCandidate>& ticl_cands,
               edm::Event& event) const override;

  private:
    std::tuple<TracksterMomentumPluginBase::LorentzVector, float> calcP4(const ticl::Trackster& trackster) const;
    bool energy_from_regression_;
    edm::EDGetTokenT<reco::TrackCollection> tracks_token_;
  };

  TracksterP4FromTrackAndPCA::TracksterP4FromTrackAndPCA(const edm::ParameterSet& ps, edm::ConsumesCollector&& ic)
      : TracksterMomentumPluginBase(ps, std::move(ic)),
        energy_from_regression_(ps.getParameter<bool>("energyFromRegression")),
        tracks_token_(ic.consumes<reco::TrackCollection>(ps.getParameter<edm::InputTag>("tracks"))) {}

  void TracksterP4FromTrackAndPCA::setP4(const std::vector<const Trackster*>& tracksters,
                                         std::vector<TICLCandidate>& ticl_cands,
                                         edm::Event& event) const {
    edm::Handle<reco::TrackCollection> tracks_h;
    event.getByToken(tracks_token_, tracks_h);
    edm::ProductID trkId = tracks_h.id();
    const reco::TrackCollection& trackCollection = *tracks_h.product();

    auto size = std::min(tracksters.size(), ticl_cands.size());
    for (size_t i = 0; i < size; ++i) {
      const auto* trackster = tracksters[i];
      // If there's a track, use it.
      if (trackster->seedIndex() != -1) {
        assert(trackster->seedID() == trkId);
        auto const& tkRef = trackCollection[trackster->seedIndex()];
        auto const& three_mom = tkRef.momentum();
        constexpr double mpion2 = 0.13957 * 0.13957;
        double energy = std::sqrt(tkRef.momentum().mag2() + mpion2);
        math::XYZTLorentzVector trk_p4(three_mom.x(), three_mom.y(), three_mom.z(), energy);
        auto& ticl_cand = ticl_cands[i];
        ticl_cand.setP4(trk_p4);
        ticl_cand.setRawEnergy(energy);
      } else {
        auto direction = trackster->eigenvectors(0).Unit();
        auto energy = energy_from_regression_ ? trackster->regressed_energy() : trackster->raw_energy();
        direction *= energy;
        math::XYZTLorentzVector cartesian(direction.X(), direction.Y(), direction.Z(), energy);
        auto& ticl_cand = ticl_cands[i];
        ticl_cand.setP4(cartesian);
        ticl_cand.setRawEnergy(energy);
      }
    }
  }
}  // namespace ticl

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(TracksterMomentumPluginFactory, ticl::TracksterP4FromTrackAndPCA, "TracksterP4FromTrackAndPCA");
