import FWCore.ParameterSet.Config as cms

genFilterEfficiencyProducer = cms.EDProducer('GenFilterEfficiencyProducer',
    filterPath = cms.string('generation_step')
)
