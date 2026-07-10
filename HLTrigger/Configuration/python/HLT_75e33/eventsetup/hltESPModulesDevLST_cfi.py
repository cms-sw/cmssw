import FWCore.ParameterSet.Config as cms

hltESPModulesDevLST = cms.ESProducer('LSTModulesDevESProducer@alpaka',
    appendToDataLabel = cms.string(''),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    )
)
