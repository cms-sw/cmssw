#ifndef MuonReco_MuonShower_h
#define MuonReco_MuonShower_h

namespace reco {
    struct MuonShower {

      /// number of muon RecHits not used by the RecSegments 
      std::vector<int> nHitsUncorrelated;
      /// the transverse size of the hit cluster
      std::vector<double> showerSizeT;
      /// the radius of the cone containing the uncorrelated hits around the track
      std::vector<double> showerDeltaR;

      MuonShower():
	nHitsUncorrelated(0),
	showerSizeT(0),showerDeltaR(0)
      { }       
    };
}
#endif

