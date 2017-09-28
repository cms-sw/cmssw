import FWCore.ParameterSet.Config as cms

simDisplacedGmtStage2Digis = cms.EDProducer(
    "L1TDisplacedMuonProducer",
    muonTag = cms.InputTag("simGmtStage2Digis"),
    emtfTag = cms.InputTag("simEmtfDigis"),
    bmtfTag = cms.InputTag("simGmtStage2Digis"),
    omtfNegTag = cms.InputTag("simGmtStage2Digis"),
    omtfPosTag = cms.InputTag("simGmtStage2Digis"),
    emtfNegTag = cms.InputTag("simGmtStage2Digis"),
    emtfPosTag = cms.InputTag("simGmtStage2Digis"),
    cscLctTag = cms.InputTag("simCscTriggerPrimitiveDigis"),
    me0TriggerTag = cms.InputTag("me0TriggerDigis"),
    me0SegmentTag = cms.InputTag("me0TriggerPseudoDigis"),
    padTag = cms.InputTag("simMuonGEMPadDigis"),
    copadTag = cms.InputTag("simCscTriggerPrimitiveDigis"),
    cscCompTag = cms.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi"),
    ## options to use extra hit information
    useGE11 = cms.bool(False),
    useGE21 = cms.bool(False),
    useME0 = cms.bool(False),
    ## 1: position based
    ## 2: direction based
    ## 3: hybrid
    method = cms.uint32(0),
    fitComparatorDigis = cms.bool(False),
    ## match to stubs
    recoverLCT = cms.bool(False),
    recoverME0 = cms.bool(False),
)

