import FWCore.ParameterSet.Config as cms

l1tStage2CaloLayer2OfflineDQM = cms.EDAnalyzer("L1TStage2CaloLayer2Offline",
    electronCollection       = cms.InputTag("gedGsfElectrons"),
    caloJetCollection        = cms.InputTag("ak4CaloJets"),
    caloMETCollection          = cms.InputTag("caloMet"),
    conversionsCollection    = cms.InputTag("allConversions"),
    PVCollection             = cms.InputTag("offlinePrimaryVerticesWithBS"),
    beamSpotCollection       = cms.InputTag("offlineBeamSpot"),

    TriggerEvent             = cms.InputTag('hltTriggerSummaryAOD','','HLT'),
    TriggerResults           = cms.InputTag('TriggerResults','','HLT'),
    #last filter of HLTEle27WP80Sequence
    TriggerFilter            = cms.InputTag('hltEle27WP80TrackIsoFilter','','HLT'),
    TriggerPath              = cms.string('HLT_Ele27_WP80_v13'),


    stage2CaloLayer2JetSource = cms.InputTag("caloStage2Digis","Jet"),
    stage2CaloLayer2EtSumSource = cms.InputTag("caloStage2Digis","EtSum"),
    
    histFolder = cms.string('L1T/L1TStage2CaloLayer2'),
)

l1tStage2CaloLayer2OfflineDQMEmu = cms.EDAnalyzer("L1TStage2CaloLayer2Offline",
    electronCollection       = cms.InputTag("gedGsfElectrons"),
    caloJetCollection        = cms.InputTag("ak4CaloJets"),
    caloMETCollection          = cms.InputTag("caloMet"),
    conversionsCollection    = cms.InputTag("allConversions"),
    PVCollection             = cms.InputTag("offlinePrimaryVerticesWithBS"),
    beamSpotCollection       = cms.InputTag("offlineBeamSpot"),

    TriggerEvent             = cms.InputTag('hltTriggerSummaryAOD','','HLT'),
    TriggerResults           = cms.InputTag('TriggerResults','','HLT'),
    #last filter of HLTEle27WP80Sequence
    TriggerFilter            = cms.InputTag('hltEle27WP80TrackIsoFilter','','HLT'),
    TriggerPath              = cms.string('HLT_Ele27_WP80_v13'),


    stage2CaloLayer2JetSource = cms.InputTag("simCaloStage2Digis"),
    stage2CaloLayer2EtSumSource = cms.InputTag("simCaloStage2Digis"),
    
    histFolder = cms.string('L1TEMU/L1TStage2CaloLayer2'),
)