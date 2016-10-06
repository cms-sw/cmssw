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
    centralBxOnly = cms.bool(False),
    ignoreHtMiss = cms.bool(False)
)

#
# Modify for running with the Stage 1 or Stage 2 trigger
#
from Configuration.Eras.Modifier_stage1L1Trigger_cff import stage1L1Trigger
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger 
_caloStage1LegacyFormatDigis = "caloStage1LegacyFormatDigis"
_params = dict(
    etTotalSource         = cms.InputTag(_caloStage1LegacyFormatDigis),
    nonIsolatedEmSource   = cms.InputTag(_caloStage1LegacyFormatDigis,"nonIsoEm"),
    etMissSource          = cms.InputTag(_caloStage1LegacyFormatDigis),
    htMissSource          = cms.InputTag(_caloStage1LegacyFormatDigis),
    forwardJetSource      = cms.InputTag(_caloStage1LegacyFormatDigis,"forJets"),
    centralJetSource      = cms.InputTag(_caloStage1LegacyFormatDigis,"cenJets"),
    tauJetSource          = cms.InputTag(_caloStage1LegacyFormatDigis,"tauJets"),
    isoTauJetSource       = cms.InputTag(_caloStage1LegacyFormatDigis,"isoTauJets"),
    isolatedEmSource      = cms.InputTag(_caloStage1LegacyFormatDigis,"isoEm"),
    etHadSource           = cms.InputTag(_caloStage1LegacyFormatDigis),
    hfRingEtSumsSource    = cms.InputTag(_caloStage1LegacyFormatDigis),
    hfRingBitCountsSource = cms.InputTag(_caloStage1LegacyFormatDigis),
    muonSource            = cms.InputTag("gtDigis"),
    centralBxOnly         = True)

stage1L1Trigger.toModify( l1extraParticles, **_params)
stage2L1Trigger.toModify( l1extraParticles, **_params)

# fastsim runs L1Reco and HLT in one step
# this requires to set :
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(l1extraParticles, centralBxOnly = True)
