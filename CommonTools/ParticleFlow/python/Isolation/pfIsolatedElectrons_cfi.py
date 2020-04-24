import FWCore.ParameterSet.Config as cms



pfIsolatedElectrons = cms.EDFilter(
    "PFCandidateFwdPtrCollectionStringFilter",
    src = cms.InputTag("pfElectronsFromVertex"),
    cut = cms.string("pt > 5 & gsfElectronRef.isAvailable() & gsfTrackRef.hitPattern().numberOfLostHits('MISSING_INNER_HITS')<2 & "\
                     "gsfElectronRef.pfIsolationVariables().sumChargedHadronPt + "\
                     "gsfElectronRef.pfIsolationVariables().sumNeutralHadronEt + "\
                     "gsfElectronRef.pfIsolationVariables().sumPhotonEt "\
                     " < 0.2 * pt "
        ),
    makeClones = cms.bool(True)
)

