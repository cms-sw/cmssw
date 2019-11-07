// Plugin setting the track pointer for TICLCandidates based on Tracksters whose seed ID and seed index refer to a
// track that is or inherits from a reco::Track.

#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoHGCal/TICL/interface/TracksterTrackPluginBase.h"

namespace ticl {
  class TracksterRecoTrackPlugin final : public TracksterTrackPluginBase {
  public:
    typedef edm::Ptr<reco::Track> TrackPtr;
    explicit TracksterRecoTrackPlugin(const edm::ParameterSet&, edm::ConsumesCollector&& iC);
    void setTrack(const std::vector<const Trackster*>& tracksters,
                  std::vector<TICLCandidate>& ticl_cands,
                  edm::Event& event) const override;
  };

  TracksterRecoTrackPlugin::TracksterRecoTrackPlugin(const edm::ParameterSet& ps, edm::ConsumesCollector&& ic)
      : TracksterTrackPluginBase(ps, std::move(ic)) {}

  void TracksterRecoTrackPlugin::setTrack(const std::vector<const Trackster*>& tracksters,
                                          std::vector<TICLCandidate>& ticl_cands,
                                          edm::Event& event) const {
    auto size = std::min(tracksters.size(), ticl_cands.size());
    for (size_t i = 0; i < size; ++i) {
      const auto& trackster = *tracksters[i];

      if (trackster.seedIndex == -1 || !trackster.seedID.isValid()) {
        return;  // leave default empty track ref
      }

      TrackPtr ptr(trackster.seedID, trackster.seedIndex, &event.productGetter());
      auto& ticl_cand = ticl_cands[i];
      ticl_cand.setTrackPtr(ptr);
    }
  }
}  // namespace ticl

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(TracksterTrackPluginFactory, ticl::TracksterRecoTrackPlugin, "TracksterRecoTrackPlugin");
