#ifndef L1Trigger_Phase2L1GMT_TPS_h
#define L1Trigger_Phase2L1GMT_TPS_h
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"
#include "L1Trigger/Phase2L1GMT/interface/TrackConverter.h"
#include "L1Trigger/Phase2L1GMT/interface/TPSAlgorithm.h"
#include "L1Trigger/Phase2L1GMT/interface/Isolation.h"

namespace Phase2L1GMT {

  class TPS {
  public:
    TPS(const edm::ParameterSet& iConfig);
    ~TPS();
    std::vector<l1t::TrackerMuon> processEvent(const std::vector<edm::Ptr<l1t::TrackerMuon::L1TTTrackType> >&,
                                               const l1t::MuonStubRefVector&);

  private:
    int verbose_;
    std::unique_ptr<TrackConverter> tt_track_converter_;
    std::unique_ptr<TPSAlgorithm> tps_;
    std::unique_ptr<Isolation> isolation_;
    std::vector<edm::Ptr<l1t::TrackerMuon::L1TTTrackType> > associateTracksWithNonant(
        const std::vector<edm::Ptr<l1t::TrackerMuon::L1TTTrackType> >& tracks, uint processor);
    l1t::SAMuonRefVector associateMuonsWithNonant(const l1t::SAMuonRefVector&, uint);
    l1t::MuonStubRefVector associateStubsWithNonant(const l1t::MuonStubRefVector&, uint);
  };
}  // namespace Phase2L1GMT

#endif
