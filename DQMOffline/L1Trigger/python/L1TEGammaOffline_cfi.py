import FWCore.ParameterSet.Config as cms

electronEfficiencyThresholds = [36, 68, 128, 176]

electronEfficiencyBins = []
electronEfficiencyBins.extend(list(xrange(0, 120, 10)))
electronEfficiencyBins.extend(list(xrange(120, 180, 20)))
electronEfficiencyBins.extend(list(xrange(180, 300, 40)))
electronEfficiencyBins.extend(list(xrange(300, 400, 100)))

l1tEGammaOfflineDQM = cms.EDAnalyzer(
    "L1TEGammaOffline",
    electronCollection=cms.InputTag("gedGsfElectrons"),
    photonCollection=cms.InputTag("photons"),
    caloJetCollection=cms.InputTag("ak4CaloJets"),
    caloMETCollection=cms.InputTag("caloMet"),
    conversionsCollection=cms.InputTag("allConversions"),
    PVCollection=cms.InputTag("offlinePrimaryVerticesWithBS"),
    beamSpotCollection=cms.InputTag("offlineBeamSpot"),

    TriggerEvent=cms.InputTag('hltTriggerSummaryAOD', '', 'HLT'),
    TriggerResults=cms.InputTag('TriggerResults', '', 'HLT'),
    # last filter of HLTEle27WP80Sequence
    TriggerFilter=cms.InputTag('hltEle27WP80TrackIsoFilter', '', 'HLT'),
    TriggerPath=cms.string('HLT_Ele27_WP80_v13'),


    stage2CaloLayer2EGammaSource=cms.InputTag("caloStage2Digis", "EGamma"),

    histFolder=cms.string('L1T/L1TEGamma'),
    electronEfficiencyThresholds=cms.vdouble(electronEfficiencyThresholds),

    electronEfficiencyBins=cms.vdouble(electronEfficiencyBins),
)

l1tEGammaOfflineDQMEmu = l1tEGammaOfflineDQM.clone(
    stage2CaloLayer2EGammaSource=cms.InputTag("simCaloStage2Digis"),

    histFolder=cms.string('L1TEMU/L1TEGamma'),
)
