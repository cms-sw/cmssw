#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoHGCal/TICL/interface/TracksterTrackPluginBase.h"

namespace ticl {
  template<class TrackClass>
  class TracksterGeneralTrackPlugin final : public TracksterTrackPluginBase {
  public:
    typedef edm::Ref<std::vector<TrackClass> > TrackRef;
    explicit TracksterGeneralTrackPlugin<TrackClass>(const edm::ParameterSet&, edm::ConsumesCollector&& iC);
    void setTrack(const ticl::Trackster& trackster, ticl::TICLCandidate& ticl_cand) const override;
  private:
    void beginEvt() override;
  };

  template<class TrackClass>
  TracksterGeneralTrackPlugin<TrackClass>::TracksterGeneralTrackPlugin(const edm::ParameterSet& ps, edm::ConsumesCollector&& ic) : 
    TracksterTrackPluginBase(ps, std::move(ic)) {// ,
  }

  template<class TrackClass>
  void TracksterGeneralTrackPlugin<TrackClass>::beginEvt() {
  }

  template<class TrackClass>
  void TracksterGeneralTrackPlugin<TrackClass>::setTrack(const ticl::Trackster& trackster, ticl::TICLCandidate& ticl_cand) const {
    if (trackster.seedIndex == 0 || !trackster.seedID.isValid()) {
      return; // leave default empty track ref
    }

    TrackRef ref(trackster.seedID, trackster.seedIndex, &this->evt().productGetter());
    ticl_cand.set_track_ref(ref);
  }

  template class TracksterGeneralTrackPlugin<reco::Track>;
  typedef TracksterGeneralTrackPlugin<reco::Track> TracksterRecoTrackPlugin;

} // namespace


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(TracksterTrackPluginFactory,
                  ticl::TracksterRecoTrackPlugin,
                  "TracksterRecoTrackPlugin");
