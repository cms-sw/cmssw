import FWCore.ParameterSet.Config as cms

L1SingleMu3 = cms.EDProducer("PATL1Producer",
    particleMaps = cms.InputTag("l1extraParticleMap","","Raw"),
    triggerName = cms.string('L1_SingleMu3'),
    objectType = cms.string('muon')
)

muonL1Producer = cms.Sequence(L1SingleMu3)

