#ifndef MuonReco_MuonIsolation_h
#define MuonReco_MuonIsolation_h


namespace reco {
   struct MuonIsolation {
     float sumPt; //!< sum-pt of tracks
     float emEt;  //!< ecal sum-Et
     float hadEt; //!< hcal sum-Et
     float hoEt;  //!< ho sum-Et
     int nTracks; //!< number of tracks in the cone (excluding veto region)
     int nJets;   //!< number of jets in the cone
     float trackerVetoPt; //!< (sum-)pt inside the veto region in r-phi
     float emVetoEt;  //!< ecal sum-et in the veto region in r-phi
     float hadVetoEt;  //!< hcal sum-et in the veto region in r-phi
     float hoVetoEt;    //!< ho sum-et in the veto region in r-phi
     MuonIsolation():
       sumPt(0),emEt(0),hadEt(0),hoEt(0),nTracks(0),nJets(0),
	  trackerVetoPt(0), emVetoEt(0), hadVetoEt(0), hoVetoEt(0){};
   };
}
#endif
