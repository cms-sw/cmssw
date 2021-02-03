import FWCore.ParameterSet.Config as cms

preshowerDetIdAssociator = cms.ESProducer("DetIdAssociatorESProducer",
    ComponentName = cms.string('PreshowerDetIdAssociator'),
    etaBinSize = cms.double(0.1),
    nEta = cms.int32(60),
    nPhi = cms.int32(30)
)
