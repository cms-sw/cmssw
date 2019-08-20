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
    void setTrack(const ticl::Trackster& trackster, ticl::TICLCandidate& ticl_cand) const override;
  private:
    void beginEvt() override;
  };

  TracksterRecoTrackPlugin::TracksterRecoTrackPlugin(const edm::ParameterSet& ps, edm::ConsumesCollector&& ic) : 
    TracksterTrackPluginBase(ps, std::move(ic)) {// ,
  }

  void TracksterRecoTrackPlugin::beginEvt() {
  }

  void TracksterRecoTrackPlugin::setTrack(const ticl::Trackster& trackster, ticl::TICLCandidate& ticl_cand) const {
    if (trackster.seedIndex == 0 || !trackster.seedID.isValid()) {
      return; // leave default empty track ref
    }

    TrackPtr ptr(trackster.seedID, trackster.seedIndex, &this->evt().productGetter());
    ticl_cand.setTrackPtr(ptr);
  }
} // namespace


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(TracksterTrackPluginFactory,
                  ticl::TracksterRecoTrackPlugin,
                  "TracksterRecoTrackPlugin");
