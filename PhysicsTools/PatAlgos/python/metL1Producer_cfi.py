import FWCore.ParameterSet.Config as cms

L1ETM20 = cms.EDProducer("PATL1Producer",
    particleMaps = cms.InputTag("l1extraParticleMap","","Raw"),
    triggerName = cms.string('L1_ETM20'),
    objectType = cms.string('met')
)

metL1Producer = cms.Sequence(L1ETM20)

