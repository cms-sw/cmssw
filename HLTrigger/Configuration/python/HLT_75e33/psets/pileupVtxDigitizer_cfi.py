import FWCore.ParameterSet.Config as cms

pileupVtxDigitizer = cms.PSet(
    accumulatorType = cms.string('PileupVertexAccumulator'),
    hitsProducer = cms.string('generator'),
    makeDigiSimLinks = cms.untracked.bool(False),
    saveVtxTimes = cms.bool(True),
    vtxFallbackTag = cms.InputTag("generator"),
    vtxTag = cms.InputTag("generatorSmeared")
)