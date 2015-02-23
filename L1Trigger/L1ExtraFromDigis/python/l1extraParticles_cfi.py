import FWCore.ParameterSet.Config as cms

l1extraParticles = cms.EDProducer("L1ExtraParticlesProd",
    muonSource = cms.InputTag("gtDigis"),
    etTotalSource = cms.InputTag("gctDigis"),
    nonIsolatedEmSource = cms.InputTag("gctDigis","nonIsoEm"),
    etMissSource = cms.InputTag("gctDigis"),
    htMissSource = cms.InputTag("gctDigis"),
    produceMuonParticles = cms.bool(True),
    forwardJetSource = cms.InputTag("gctDigis","forJets"),
    centralJetSource = cms.InputTag("gctDigis","cenJets"),
    produceCaloParticles = cms.bool(True),
    tauJetSource = cms.InputTag("gctDigis","tauJets"),
    isoTauJetSource = cms.InputTag("gctDigis","isoTauJets"),
    isolatedEmSource = cms.InputTag("gctDigis","isoEm"),
    etHadSource = cms.InputTag("gctDigis"),
    hfRingEtSumsSource = cms.InputTag("gctDigis"),
    hfRingBitCountsSource = cms.InputTag("gctDigis"),
    centralBxOnly = cms.bool(True),
    ignoreHtMiss = cms.bool(False)
)

#
# Modify for running with the Stage 1 trigger
#
from Configuration.StandardSequences.Eras import eras
eras.stage1L1Trigger.toModify( l1extraParticles, etTotalSource = cms.InputTag("caloStage1LegacyFormatDigis") )
eras.stage1L1Trigger.toModify( l1extraParticles, nonIsolatedEmSource = cms.InputTag("caloStage1LegacyFormatDigis","nonIsoEm") )
eras.stage1L1Trigger.toModify( l1extraParticles, etMissSource = cms.InputTag("caloStage1LegacyFormatDigis") )
eras.stage1L1Trigger.toModify( l1extraParticles, htMissSource = cms.InputTag("caloStage1LegacyFormatDigis") )
eras.stage1L1Trigger.toModify( l1extraParticles, forwardJetSource = cms.InputTag("caloStage1LegacyFormatDigis","forJets") )
eras.stage1L1Trigger.toModify( l1extraParticles, centralJetSource = cms.InputTag("caloStage1LegacyFormatDigis","cenJets") )
eras.stage1L1Trigger.toModify( l1extraParticles, tauJetSource = cms.InputTag("caloStage1LegacyFormatDigis","tauJets") )
eras.stage1L1Trigger.toModify( l1extraParticles, isoTauJetSource = cms.InputTag("caloStage1LegacyFormatDigis","isoTauJets") )
eras.stage1L1Trigger.toModify( l1extraParticles, isolatedEmSource = cms.InputTag("caloStage1LegacyFormatDigis","isoEm") )
eras.stage1L1Trigger.toModify( l1extraParticles, etHadSource = cms.InputTag("caloStage1LegacyFormatDigis") )
eras.stage1L1Trigger.toModify( l1extraParticles, hfRingEtSumsSource = cms.InputTag("caloStage1LegacyFormatDigis") )
eras.stage1L1Trigger.toModify( l1extraParticles, hfRingBitCountsSource = cms.InputTag("caloStage1LegacyFormatDigis") )
