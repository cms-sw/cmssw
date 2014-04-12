import FWCore.ParameterSet.Config as cms


pfIsolatedMuons = cms.EDFilter(
    "PFCandidateFwdPtrCollectionStringFilter",
    src = cms.InputTag("pfMuonsFromVertex"),
    cut = cms.string("pt > 5 & muonRef.isAvailable() & "\
                     "muonRef.pfIsolationR04().sumChargedHadronPt + "\
                     "muonRef.pfIsolationR04().sumNeutralHadronEt + "\
                     "muonRef.pfIsolationR04().sumPhotonEt "\
                     " < 0.15 * pt "
        ),
    makeClones = cms.bool(True)
)

