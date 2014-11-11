import FWCore.ParameterSet.Config as cms

DQMExample_Step1 = cms.EDAnalyzer("DQMExample_Step1",
    electronCollection       = cms.InputTag("gsfElectrons"),
    caloJetCollection        = cms.InputTag("ak5CaloJets"),
    pfMETCollection          = cms.InputTag("pfMet"),
    conversionsCollection    = cms.InputTag("allConversions"),
    PVCollection             = cms.InputTag("offlinePrimaryVerticesWithBS"),
    beamSpotCollection       = cms.InputTag("offlineBeamSpot"),

    TriggerEvent             = cms.InputTag('hltTriggerSummaryAOD','','HLT'),
    TriggerResults           = cms.InputTag('TriggerResults','','HLT'),
    #last filter of HLTEle27WP80Sequence
    TriggerFilter            = cms.InputTag('hltEle27WP80TrackIsoFilter','','HLT'),
    TriggerPath              = cms.string('HLT_Ele27_WP80_v13'),


    PtThrL1 = cms.untracked.double(30.0),
    PtThrL2 = cms.untracked.double(10.0),
    PtThrJet = cms.untracked.double(20.0),
    PtThrMet = cms.untracked.double(20.0),

)

