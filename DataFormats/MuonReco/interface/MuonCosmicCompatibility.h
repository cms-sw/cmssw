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
      /// cosmic-likeness based on the 2D impact parameters (dxy, dz wrt to PV). 0 == cosmic-like
      float ipCompatibility;
      /// cosmic-likeness based on the event activity information: tracker track multiplicity and vertex quality. 0 == cosmic-like
      float vertexCompatibility;

      MuonCosmicCompatibility():
	cosmicCompatibility(0),timeCompatibility(0),
	backToBackCompatibility(0),overlapCompatibility(0),
        ipCompatibility(0), vertexCompatibility(0)
      { }       
    };
}
#endif
