import FWCore.ParameterSet.Config as cms
from DQMOffline.L1Trigger.L1THistDefinitions_cff import histDefinitions

tauEfficiencyThresholds = [28, 30, 32, 128, 176]

tauEfficiencyBins = []
tauEfficiencyBins.extend(list(range(0, 50, 1)))
tauEfficiencyBins.extend(list(range(50, 60, 2)))
tauEfficiencyBins.extend(list(range(60, 80, 5)))
tauEfficiencyBins.extend(list(range(80, 200, 10)))
tauEfficiencyBins.extend(list(range(200, 300, 20)))
tauEfficiencyBins.extend(list(range(300, 400, 50)))
tauEfficiencyBins.extend(list(range(400, 600, 100)))
tauEfficiencyBins.extend(list(range(600, 1200, 200)))

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tTauOfflineDQM = DQMEDAnalyzer(
    "L1TTauOffline",
    verbose   = cms.untracked.bool(False),

    muonInputTag = cms.untracked.InputTag("muons"),
    tauInputTag = cms.untracked.InputTag("hpsPFTauProducer"),
    metInputTag = cms.untracked.InputTag("pfMet"),
    antiMuInputTag = cms.untracked.InputTag("hpsPFTauDiscriminationByMuonRejection3"),
    antiMuWP = cms.untracked.string("ByTightMuonRejection3"),
    antiEleInputTag = cms.untracked.InputTag("hpsPFTauDiscriminationByMVA6ElectronRejection"),
    antiEleWP = cms.untracked.string("_Loose"),
    decayModeFindingInputTag = cms.untracked.InputTag("hpsPFTauDiscriminationByDecayModeFindingOldDMs"),
    comb3TInputTag = cms.untracked.InputTag("hpsPFTauBasicDiscriminators"),
    comb3TWP = cms.untracked.string("ByTightCombinedIsolationDBSumPtCorr3Hits"),
    l1tInputTag  = cms.untracked.InputTag("caloStage2Digis:Tau"),
    vtxInputTag = cms.untracked.InputTag("offlinePrimaryVertices"),
    bsInputTag  = cms.untracked.InputTag("offlineBeamSpot"),
    triggerNames = cms.untracked.vstring("HLT_IsoMu18_v*","HLT_IsoMu20_v*","HLT_IsoMu22_v*","HLT_IsoMu24_v*","HLT_IsoMu27_v*", "HLT_IsoMu30_v*"),
    trigInputTag       = cms.untracked.InputTag("hltTriggerSummaryAOD", "", "HLT"),
    trigProcess        = cms.untracked.string("HLT"),
    trigProcess_token  = cms.untracked.InputTag("TriggerResults","","HLT"),

    stage2CaloLayer2TauSource=cms.InputTag("simCaloStage2Digis"),
    histFolder=cms.string('L1T/L1TObjects/L1TTau/L1TriggerVsReco'),

    tauEfficiencyThresholds=cms.vint32(tauEfficiencyThresholds),
    tauEfficiencyBins=cms.vdouble(tauEfficiencyBins),

    histDefinitions=cms.PSet(
        nVertex=histDefinitions.nVertex.clone(),
        ETvsET=histDefinitions.ETvsET.clone(),
        PHIvsPHI=histDefinitions.PHIvsPHI.clone(),
    ),

)

l1tTauOfflineDQMEmu = l1tTauOfflineDQM.clone(
    stage2CaloLayer2TauSource= "simCaloStage2Digis",

    histFolder= 'L1TEMU/L1TObjects/L1TTau/L1TriggerVsReco'
)
