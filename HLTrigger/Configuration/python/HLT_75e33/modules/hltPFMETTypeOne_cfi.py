import FWCore.ParameterSet.Config as cms

hltPFMETTypeOne = cms.EDProducer("CorrectedPFMETProducer",
    src = cms.InputTag("hltPFMET"),
    srcCorrections = cms.VInputTag("hltPFMETTypeOneCorrector:type1")
)
