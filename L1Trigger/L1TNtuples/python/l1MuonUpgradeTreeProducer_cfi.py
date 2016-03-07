import FWCore.ParameterSet.Config as cms

l1MuonUpgradeTreeProducer = cms.EDAnalyzer("L1MuonUpgradeTreeProducer",
  ugmtTag = cms.InputTag("simGmtDigis"),
  bmtfTag = cms.InputTag("simBmtfDigis", "BMTF"),
  omtfTag = cms.InputTag("simOmtfDigis", "OMTF"),
  emtfTag = cms.InputTag("simEmtfDigis", "EMTF"),
  calo2x2Tag = cms.InputTag("simGmtCaloSumDigis", "TriggerTower2x2s"),
  caloTag = cms.InputTag("simGmtCaloSumDigis"),
  caloRecoTag = cms.InputTag("none")
)
