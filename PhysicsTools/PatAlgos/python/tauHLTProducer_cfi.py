import FWCore.ParameterSet.Config as cms

HLT1Tau = cms.EDProducer("PATHLTProducer",
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    filterName = cms.InputTag("filterL3SingleTau","","HLT"),
    triggerName = cms.string('HLT1Tau')
)

HLT2TauPixel = cms.EDProducer("PATHLTProducer",
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    filterName = cms.InputTag("filterL25PixelTau","","HLT"),
    triggerName = cms.string('HLT2TauPixel')
)

tauHLTProducer = cms.Sequence(HLT1Tau*HLT2TauPixel)

