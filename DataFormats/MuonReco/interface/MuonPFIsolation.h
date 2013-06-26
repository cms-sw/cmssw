#ifndef MuonReco_MuonPFIsolation_h
#define MuonReco_MuonPFIsolation_h


namespace reco {
   struct MuonPFIsolation {
     float sumChargedHadronPt; //!< sum-pt of charged Hadron 
     float sumChargedParticlePt; //!< sum-pt of charged Particles(inludes e/mu) 
     float sumNeutralHadronEt;  //!< sum pt of neutral hadrons
     float sumPhotonEt;  //!< sum pt of PF photons
     float sumNeutralHadronEtHighThreshold;  //!< sum pt of neutral hadrons with a higher threshold
     float sumPhotonEtHighThreshold;  //!< sum pt of PF photons with a higher threshold
     float sumPUPt;  //!< sum pt of charged Particles not from PV  (for Pu corrections)
     
     MuonPFIsolation():
       sumChargedHadronPt(0),sumChargedParticlePt(0),sumNeutralHadronEt(0),sumPhotonEt(0),
       sumNeutralHadronEtHighThreshold(0), sumPhotonEtHighThreshold(0), sumPUPt(0) {}

   };


}
#endif
