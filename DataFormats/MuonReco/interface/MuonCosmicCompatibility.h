#ifndef MuonReco_MuonCosmicCompatibility_h
#define MuonReco_MuonCosmicCompatibility_h

namespace reco {
    struct MuonCosmicCompatibility {

      /// combined cosmic-likeness: 0 == not cosmic-like
      float cosmicCompatibility;
      /// cosmic-likeness based on time: 0 == prompt-like
      float timeCompatibility;
      /// cosmic-likeness based on presence of a track in opp side: 0 == no matching opp tracks
      float backToBackCompatibility;
      /// cosmic-likeness based on overlap with traversing cosmic muon (only muon/STA hits are used)
      float overlapCompatibility;

      MuonCosmicCompatibility():
	cosmicCompatibility(0),timeCompatibility(0),
	backToBackCompatibility(0),overlapCompatibility(0)
      { }       
    };
}
#endif
