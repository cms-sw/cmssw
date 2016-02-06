import FWCore.ParameterSet.Config as cms

#
# Muons not-included yet... need legacy to upgrade format converter
#

l1tSummaryStage1Digis = cms.EDAnalyzer(
    "L1TSummary",
    tag       = cms.string("Stage 1 Digis"),
    egCheck   = cms.bool(True),
    tauCheck  = cms.bool(True),
    jetCheck  = cms.bool(True),
    sumCheck  = cms.bool(True),
    muonCheck = cms.bool(False),
    egToken   = cms.InputTag("caloStage1FinalDigis"),
    tauToken  = cms.InputTag("caloStage1FinalDigis"),
    jetToken  = cms.InputTag("caloStage1FinalDigis"),
    sumToken  = cms.InputTag("caloStage1FinaleDigis"),
    #muonToken = cms.InputTag("gmtStage2Digis",""),
)
