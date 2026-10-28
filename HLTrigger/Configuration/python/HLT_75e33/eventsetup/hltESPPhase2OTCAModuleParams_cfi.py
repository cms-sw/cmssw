import FWCore.ParameterSet.Config as cms

hltESPPhase2OTCAModuleParams = cms.ESProducer('Phase2OTCAModuleParamsESProducerAlpaka@alpaka',
    Phase2StripCPE = cms.ESInputTag('hltESPPhase2StripCPE', 'Phase2StripCPEHLT'),
    appendToDataLabel = cms.string('HLT'),
    alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)
