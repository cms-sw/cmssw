#ifndef MuonReco_MuonShower_h
#define MuonReco_MuonShower_h

#include <vector>

namespace reco {
  struct MuonShower {
    /// number of all the muon RecHits per chamber crossed by a track (1D hits)
    std::vector<int> nStationHits;
    /// number of the muon RecHits used by segments per chamber crossed by a track
    std::vector<int> nStationCorrelatedHits;
    /// the transverse size of the hit cluster
    std::vector<float> stationShowerSizeT;
    /// the radius of the cone containing the all the hits around the track
    std::vector<float> stationShowerDeltaR;

    MuonShower() : nStationHits(0), nStationCorrelatedHits(0), stationShowerSizeT(0), stationShowerDeltaR(0) {}
  };
}  // namespace reco
#endif
