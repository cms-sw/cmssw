import FWCore.ParameterSet.Config as cms

hltPFPuppiMETTypeOne = cms.EDProducer("CorrectedPFMETProducer",
    src = cms.InputTag("hltPFPuppiMET"),
    srcCorrections = cms.VInputTag("hltPFPuppiMETTypeOneCorrector:type1")
)
# foo bar baz
# 7xNglJqHases1
# 6h2zi9aaqXA5C
