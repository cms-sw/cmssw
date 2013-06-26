import FWCore.ParameterSet.Config as cms

pfAllPhotons = cms.EDFilter("PFCandidateFwdPtrCollectionPdgIdFilter",
    src = cms.InputTag("pfNoPileUpIso"),
    pdgId = cms.vint32(22),
    makeClones = cms.bool(True)
)



