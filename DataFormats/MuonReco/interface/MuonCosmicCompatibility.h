#ifndef MuonReco_MuonCosmicCompatibility_h
#define MuonReco_MuonCosmicCompatibility_h

namespace reco {
    struct MuonCosmicCompatibility {
      ///
      /// bool returns true if standAloneMuon_updatedAtVtx was used in the fit
      float cosmicCompatibility;
      /// value of the kink algorithm applied to the inner track stub
      float timeCompatibility;
      /// value of the kink algorithm applied to the global track
      float backToBackCompatibility;
      /// chi2 value for the inner track stub with respect to the global track
      float overlapCompatibility;

      MuonCosmicCompatibility():
	cosmicCompatibility(0),timeCompatibility(0),
	backToBackCompatibility(0),overlapCompatibility(0)
      { }       
    };
}
#endif
