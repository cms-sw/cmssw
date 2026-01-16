import FWCore.ParameterSet.Config as cms

hltESPPixelCPEFastParamsPhase2 = cms.ESProducer('PixelCPEFastParamsESProducerAlpakaPhase2@alpaka', 
    ComponentName = cms.string("PixelCPEFastParamsPhase2"),
    appendToDataLabel = cms.string(''),
    alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)
