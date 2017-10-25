import FWCore.ParameterSet.Config as cms

tauEfficiencyThresholds = [28, 30, 32, 128, 176]

tauEfficiencyBins = []
tauEfficiencyBins.extend(list(xrange(0, 120, 1)))
tauEfficiencyBins.extend(list(xrange(120, 180, 20)))
tauEfficiencyBins.extend(list(xrange(180, 300, 40)))
tauEfficiencyBins.extend(list(xrange(300, 401, 100)))

l1tTauOfflineDQM = cms.EDAnalyzer(
    "L1TTauOffline",
    verbose   = cms.untracked.bool(False),

    muonInputTag = cms.untracked.InputTag("muons"),
    tauInputTag = cms.untracked.InputTag("hpsPFTauProducer"),
    metInputTag = cms.untracked.InputTag("pfMet"),
    antiMuInputTag = cms.untracked.InputTag("hpsPFTauDiscriminationByTightMuonRejection3"),
    antiEleInputTag = cms.untracked.InputTag("hpsPFTauDiscriminationByMVA6LooseElectronRejection"),
    decayModeFindingInputTag = cms.untracked.InputTag("hpsPFTauDiscriminationByDecayModeFindingOldDMs"),
    comb3TInputTag = cms.untracked.InputTag("hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits"),
    l1tInputTag  = cms.untracked.InputTag("caloStage2Digis:Tau"),
    vtxInputTag = cms.untracked.InputTag("offlinePrimaryVertices"),
    bsInputTag  = cms.untracked.InputTag("offlineBeamSpot"),
    triggerNames = cms.untracked.vstring("HLT_IsoMu18_v*","HLT_IsoMu20_v*","HLT_IsoMu22_v*","HLT_IsoMu24_v*","HLT_IsoMu27_v*"),
    trigInputTag       = cms.untracked.InputTag("hltTriggerSummaryAOD", "", "HLT"),
    trigProcess        = cms.untracked.string("HLT"),
    trigProcess_token  = cms.untracked.InputTag("TriggerResults","","HLT"),

    histFolder=cms.string('L1T/L1TTau'),

    tauEfficiencyThresholds=cms.vint32(tauEfficiencyThresholds),
    tauEfficiencyBins=cms.vdouble(tauEfficiencyBins),

)

l1tTauOfflineDQMEmu = l1tTauOfflineDQM.clone(
    stage2CaloLayer2TauSource=cms.InputTag("simCaloStage2Digis"),

    histFolder=cms.string('L1TEMU/L1TTau'),
)
