#ifndef MuonReco_MuonIsolation_h
#define MuonReco_MuonIsolation_h


namespace reco {
   struct MuonIsolation {
      float sumPt;
      float emEnergy;
      float hadEnergy;
      float hoEnergy;
      int nTracks;
      int nJets;
      MuonIsolation():
      sumPt(0),emEnergy(0),hadEnergy(0),hoEnergy(0),nTracks(0),nJets(0){};
   };
}
#endif
