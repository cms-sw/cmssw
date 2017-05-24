import FWCore.ParameterSet.Config as cms

tauEfficiencyThresholds = [20, 32, 128, 176]
#tauEfficiencyThresholds = [30, 32, 128, 176]

tauEfficiencyBins = []
tauEfficiencyBins.extend(list(xrange(0, 120, 1)))
tauEfficiencyBins.extend(list(xrange(120, 180, 20)))
tauEfficiencyBins.extend(list(xrange(180, 300, 40)))
tauEfficiencyBins.extend(list(xrange(300, 400, 100)))

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
    
    #triggerNames = cms.untracked.vstring("HLT_IsoMu22_v*"),
    triggerNames = cms.untracked.vstring("HLT_IsoMu18_v*","HLT_IsoMu20_v*","HLT_IsoMu22_v*","HLT_IsoMu24_v*","HLT_IsoMu27_v*"),
    #triggerNames = cms.untracked.vstring("HLT_IsoMu24_v*"),

    trigInputTag       = cms.untracked.InputTag("hltTriggerSummaryAOD", "", "HLT"),
    trigProcess        = cms.untracked.string("HLT"),
    trigProcess_token  = cms.untracked.InputTag("TriggerResults","","HLT"),
    

    #electronCollection=cms.InputTag("gedGsfElectrons"),
    #photonCollection=cms.InputTag("photons"),
    #caloJetCollection=cms.InputTag("ak4CaloJets"),
    #caloMETCollection=cms.InputTag("caloMet"),
    #conversionsCollection=cms.InputTag("allConversions"),
    #PVCollection=cms.InputTag("offlinePrimaryVerticesWithBS"),
    #beamSpotCollection=cms.InputTag("offlineBeamSpot"),

    #TriggerEvent=cms.InputTag('hltTriggerSummaryAOD', '', 'HLT'),
    #TriggerResults=cms.InputTag('TriggerResults', '', 'HLT'),
    # last filter of HLTEle27WP80Sequence
    #TriggerFilter=cms.InputTag('hltEle27WP80TrackIsoFilter', '', 'HLT'),
    #TriggerPath=cms.string('HLT_Ele27_WP80_v13'),


    #stage2CaloLayer2TauSource=cms.InputTag("caloStage2Digis", "Tau"),

    histFolder=cms.string('L1T/L1TTau'),

    tauEfficiencyThresholds=cms.vdouble(tauEfficiencyThresholds),
    tauEfficiencyBins=cms.vdouble(tauEfficiencyBins),
    
)

l1tTauOfflineDQMEmu = l1tTauOfflineDQM.clone(
    stage2CaloLayer2TauSource=cms.InputTag("simCaloStage2Digis"),

    histFolder=cms.string('L1TEMU/L1TTau'),
)
