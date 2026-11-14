import FWCore.ParameterSet.Config as cms

hcalDetIdAssociator = cms.ESProducer("DetIdAssociatorESProducer",
    ComponentName = cms.string('HcalDetIdAssociator'),
    etaBinSize = cms.double(0.087),
    hcalRegion = cms.int32(1),
    nEta = cms.int32(70),
    nPhi = cms.int32(72)
)
