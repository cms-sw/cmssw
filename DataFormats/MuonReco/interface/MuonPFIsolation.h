#ifndef MuonReco_MuonPFIsolation_h
#define MuonReco_MuonPFIsolation_h


namespace reco {
   struct MuonPFIsolation {
     float sumChargedHadronPt; //!< sum-pt of charged Hadron 
     float sumChargedParticlePt; //!< sum-pt of charged Particles(inludes e/mu) 
     float sumNeutralHadronEt;  //!< sum pt of neutral hadrons
     float sumPhotonEt;  //!< sum pt of PF photons
     float sumPUPt;  //!< sum pt of charged Particles not from PV  (for Pu corrections)
     MuonPFIsolation():
       sumChargedHadronPt(0),sumChargedParticlePt(0),sumNeutralHadronEt(0),sumPhotonEt(0),sumPUPt(0) {};
   };


}
#endif
