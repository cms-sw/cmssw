import FWCore.ParameterSet.Config as cms

hltPFPuppiMETTypeOne = cms.EDProducer("CorrectedPFMETProducer",
    src = cms.InputTag("hltPFPuppiMET"),
    srcCorrections = cms.VInputTag("hltPFPuppiMETTypeOneCorrector:type1")
)
