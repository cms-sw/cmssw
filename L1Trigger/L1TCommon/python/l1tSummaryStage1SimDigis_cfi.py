import FWCore.ParameterSet.Config as cms

#
# Muons not-included yet... need legacy to upgrade format converter
#

l1tSummaryStage1SimDigis = cms.EDAnalyzer(
    "L1TSummary",
    tag        = cms.string("Stage 1 Simulated Digis"),
    egCheck    = cms.bool(True),
    tauCheck   = cms.bool(True),
    jetCheck   = cms.bool(True),
    sumCheck   = cms.bool(True),
    muonCheck  = cms.bool(False),
    bxZeroOnly = cms.bool(True),
    egToken    = cms.InputTag("simCaloStage1FinalDigis"),
    tauTokens  = cms.VInputTag(["simCaloStage1FinalDigis:rlxTaus","simCaloStage1FinalDigis:isoTaus"]),
    jetToken   = cms.InputTag("simCaloStage1FinalDigis"),
    sumToken   = cms.InputTag("simCaloStage1FinalDigis"),
    #muonToken = cms.InputTag("simGmtStage2Digis",""),  # no stage-1 analog yet...
)
