import FWCore.ParameterSet.Config as cms

muonDetIdAssociator = cms.ESProducer("DetIdAssociatorESProducer",
    ComponentName = cms.string('MuonDetIdAssociator'),
    etaBinSize = cms.double(0.125),
    includeBadChambers = cms.bool(True),
    includeGEM = cms.bool(True),
    includeME0 = cms.bool(True),
    nEta = cms.int32(48),
    nPhi = cms.int32(48)
)
