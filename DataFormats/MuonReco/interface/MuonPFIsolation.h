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
     float radialChargedHadronIso; //!< cut based radial isolation variable
     float radialChargedParticleIso; //!< cut based radial isolation variable
     float radialPhotonIso; //!< cut based radial isolation variable
     float radialNeutralHadronIso; //!< cut based radial isolation variable
     float sumChargedHadronDR; //!< Sum DR  of charged hadrons .To be used for MVA radial isolation
     float meanChargedHadronDR; //!< Mean DR of charged hadrons. To Be used for MVA 

     MuonPFIsolation():
       sumChargedHadronPt(0),sumChargedParticlePt(0),sumNeutralHadronEt(0),sumPhotonEt(0),
       sumNeutralHadronEtHighThreshold(0), sumPhotonEtHighThreshold(0), sumPUPt(0), radialChargedHadronIso(0),
       radialChargedParticleIso(0),radialPhotonIso(0),radialNeutralHadronIso(0),
       sumChargedHadronDR(0),meanChargedHadronDR(0) {};
   };


}
#endif
