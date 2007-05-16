#ifndef MuonReco_MuonIsolation_h
#define MuonReco_MuonIsolation_h


namespace reco {
   struct MuonIsolation {
      float sumPt;
      float emEt;
      float hadEt;
      float hoEt;
      int nTracks;
      int nJets;
      MuonIsolation():
      sumPt(0),emEt(0),hadEt(0),hoEt(0),nTracks(0),nJets(0){};
   };
}
#endif
