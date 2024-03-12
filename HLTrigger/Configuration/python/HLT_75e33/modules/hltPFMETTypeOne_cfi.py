import FWCore.ParameterSet.Config as cms

hltPFMETTypeOne = cms.EDProducer("CorrectedPFMETProducer",
    src = cms.InputTag("hltPFMET"),
    srcCorrections = cms.VInputTag("hltPFMETTypeOneCorrector:type1")
)
# foo bar baz
# 7QfNRkJgy7tF7
