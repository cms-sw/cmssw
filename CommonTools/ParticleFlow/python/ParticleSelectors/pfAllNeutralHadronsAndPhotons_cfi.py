import FWCore.ParameterSet.Config as cms

pfAllNeutralHadronsAndPhotons = cms.EDFilter("PFCandidateFwdPtrCollectionPdgIdFilter",
    src = cms.InputTag("pfNoPileUpIso"),
    pdgId = cms.vint32(22,111,130,310,2112),
    makeClones = cms.bool(True)
)



