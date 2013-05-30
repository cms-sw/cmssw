import FWCore.ParameterSet.Config as cms



pfIsolatedElectrons = cms.EDFilter(
    "PFCandidateFwdPtrCollectionStringFilter",
    src = cms.InputTag("pfElectronsFromVertex"),
    cut = cms.string(" pt > 5 & gsfElectronRef.isAvailable() & gsfTrackRef.trackerExpectedHitsInner.numberOfLostHits<2 & "\
                     "gsfElectronRef.pfIsolationVariables().chargedHadronIso + "\
                     "gsfElectronRef.pfIsolationVariables().neutralHadronIso + "\
                     "gsfElectronRef.pfIsolationVariables().photonIso "\
                     " < 0.2 * pt "
        ),
    makeClones = cms.bool(True)
)

