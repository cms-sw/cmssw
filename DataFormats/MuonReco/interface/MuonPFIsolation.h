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

     float meanChargedHadronDR; //!< mean charged Hadron DR -for radial isolation
     float sumChargedHadronDR;  //!< sum charged Hadron DR -for radial isolation

     float meanChargedParticleDR;//!< mean charged particle DR -for radial isolation
     float sumChargedParticleDR; //!< sum charged particle DR -for radial isolation

     float meanPhotonDR;//!< mean photon  DR -for radial isolation
     float sumPhotonDR;//!< sum photon  DR -for radial isolation

     float meanNeutralHadronDR;//!< mean neutral hadron  DR -for radial isolation
     float sumNeutralHadronDR; //!< sum neutral hadron  DR -for radial isolation

     float meanPhotonDRHighThreshold;//!< mean photon  DR tighter threshold-for radial isolation
     float sumPhotonDRHighThreshold; //!< sum photon  DR tighter threshold-for radial isolation

     float meanNeutralHadronDRHighThreshold;//!< mean neutralHadron  DR tighter threshold-for radial isolation
     float sumNeutralHadronDRHighThreshold;//!< sum neutralHadron  DR tighter threshold-for radial isolation

     float meanPUDR;//!< mean PU   DR -for radial isolation
     float sumPUDR;//!< sum PU  DR -for radial isolation
     
     MuonPFIsolation():
       sumChargedHadronPt(0),sumChargedParticlePt(0),sumNeutralHadronEt(0),sumPhotonEt(0),
       sumNeutralHadronEtHighThreshold(0), sumPhotonEtHighThreshold(0), sumPUPt(0),
       meanChargedHadronDR(0),sumChargedHadronDR(0),meanChargedParticleDR(0),sumChargedParticleDR(0),
       meanPhotonDR(0),sumPhotonDR(0),meanNeutralHadronDR(0),sumNeutralHadronDR(0),meanPhotonDRHighThreshold(0),
       sumPhotonDRHighThreshold(0),meanNeutralHadronDRHighThreshold(0),sumNeutralHadronDRHighThreshold(0),
       meanPUDR(0),sumPUDR(0) {}

   };


}
#endif
