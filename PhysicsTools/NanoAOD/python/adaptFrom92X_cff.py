import FWCore.ParameterSet.Config as cms

slimmedMuons = cms.EDProducer("PATMuonUpdater",
    src = cms.InputTag("slimmedMuons", processName = cms.InputTag.skipCurrentProcess()),
    vertices = cms.InputTag("offlineSlimmedPrimaryVertices")
)
slimmedElectrons = cms.EDProducer("PATElectronUpdater",
    src = cms.InputTag("slimmedElectrons", processName = cms.InputTag.skipCurrentProcess()),
    vertices = cms.InputTag("offlineSlimmedPrimaryVertices")
)

adapt_nano = cms.Sequence( slimmedMuons + slimmedElectrons )
