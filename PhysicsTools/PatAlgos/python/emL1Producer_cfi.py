import FWCore.ParameterSet.Config as cms

L1SingleEG5 = cms.EDProducer("PATL1Producer",
    particleMaps = cms.InputTag("l1extraParticleMap","","Raw"),
    triggerName = cms.string('L1_SingleEG5'),
    objectType = cms.string('em')
)

emL1Producer = cms.Sequence(L1SingleEG5)

