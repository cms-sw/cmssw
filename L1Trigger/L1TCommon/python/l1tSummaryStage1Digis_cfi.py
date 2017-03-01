import FWCore.ParameterSet.Config as cms

#
# Muons not-included yet... need legacy to upgrade format converter
#

l1tSummaryStage1Digis = cms.EDAnalyzer(
    "L1TSummary",
    tag        = cms.string("Stage 1 Unpacked Digis"),
    egCheck    = cms.bool(True),
    tauCheck   = cms.bool(True),
    jetCheck   = cms.bool(True),
    sumCheck   = cms.bool(True),
    muonCheck  = cms.bool(False),
    bxZeroOnly = cms.bool(True),
    egToken    = cms.InputTag("caloStage1FinalDigis"),
    tauTokens  = cms.VInputTag(["caloStage1FinalDigis:rlxTaus","caloStage1FinalDigis:isoTaus"]),
    jetToken   = cms.InputTag("caloStage1FinalDigis"),
    sumToken   = cms.InputTag("caloStage1FinalDigis"),
    #muonToken = cms.InputTag(""),
)
