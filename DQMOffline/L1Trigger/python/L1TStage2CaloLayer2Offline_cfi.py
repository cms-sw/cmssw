import FWCore.ParameterSet.Config as cms

jetEfficiencyThresholds = [36, 68, 128, 176]
metEfficiencyThresholds = [40, 60, 80, 100, 120]
mhtEfficiencyThresholds = [40, 60, 80, 100, 120]
ettEfficiencyThresholds = [30, 50, 90, 140]
httEfficiencyThresholds = [120, 160, 200, 240, 280]

jetEfficiencyBins = []
jetEfficiencyBins.extend(list(xrange(0, 120, 10)))
jetEfficiencyBins.extend(list(xrange(120, 180, 20)))
jetEfficiencyBins.extend(list(xrange(180, 300, 40)))
jetEfficiencyBins.extend(list(xrange(300, 401, 100)))

metEfficiencyBins = []
metEfficiencyBins.extend(list(xrange(0, 40, 4)))
metEfficiencyBins.extend(list(xrange(40, 70, 2)))
metEfficiencyBins.extend(list(xrange(70, 100, 5)))
metEfficiencyBins.extend(list(xrange(100, 160, 10)))
metEfficiencyBins.extend(list(xrange(160, 261, 20)))

mhtEfficiencyBins = []
mhtEfficiencyBins.extend(list(xrange(30, 50, 1)))
mhtEfficiencyBins.extend(list(xrange(50, 80, 5)))
mhtEfficiencyBins.extend(list(xrange(80, 140, 10)))
mhtEfficiencyBins.extend(list(xrange(140, 200, 15)))
mhtEfficiencyBins.extend(list(xrange(200, 300, 20)))
mhtEfficiencyBins.extend(list(xrange(300, 401, 50)))

ettEfficiencyBins = []
ettEfficiencyBins.extend(list(xrange(0, 30, 30)))
ettEfficiencyBins.extend(list(xrange(30, 50, 10)))
ettEfficiencyBins.extend(list(xrange(50, 90, 5)))
ettEfficiencyBins.extend(list(xrange(90, 141, 2)))

httEfficiencyBins = []
httEfficiencyBins.extend(list(xrange(0, 100, 5)))
httEfficiencyBins.extend(list(xrange(100, 200, 10)))
httEfficiencyBins.extend(list(xrange(200, 400, 20)))
httEfficiencyBins.extend(list(xrange(400, 500, 50)))
httEfficiencyBins.extend(list(xrange(500, 601, 10)))

l1tStage2CaloLayer2OfflineDQM = cms.EDAnalyzer(
    "L1TStage2CaloLayer2Offline",
    electronCollection=cms.InputTag("gedGsfElectrons"),
    caloJetCollection=cms.InputTag("ak4CaloJets"),
    caloMETCollection=cms.InputTag("caloMetBE"),
    # MET collection including HF
    caloETMHFCollection=cms.InputTag("caloMet"),
    conversionsCollection=cms.InputTag("allConversions"),
    PVCollection=cms.InputTag("offlinePrimaryVerticesWithBS"),
    beamSpotCollection=cms.InputTag("offlineBeamSpot"),

    TriggerEvent=cms.InputTag('hltTriggerSummaryAOD', '', 'HLT'),
    TriggerResults=cms.InputTag('TriggerResults', '', 'HLT'),
    # last filter of HLTEle27WP80Sequence
    TriggerFilter=cms.InputTag('hltEle27WP80TrackIsoFilter', '', 'HLT'),
    TriggerPath=cms.string('HLT_Ele27_WP80_v13'),


    stage2CaloLayer2JetSource=cms.InputTag("caloStage2Digis", "Jet"),
    stage2CaloLayer2EtSumSource=cms.InputTag("caloStage2Digis", "EtSum"),

    histFolder=cms.string('L1T/L1TStage2CaloLayer2'),
    jetEfficiencyThresholds=cms.vdouble(jetEfficiencyThresholds),
    metEfficiencyThresholds=cms.vdouble(metEfficiencyThresholds),
    mhtEfficiencyThresholds=cms.vdouble(mhtEfficiencyThresholds),
    ettEfficiencyThresholds=cms.vdouble(ettEfficiencyThresholds),
    httEfficiencyThresholds=cms.vdouble(httEfficiencyThresholds),

    jetEfficiencyBins=cms.vdouble(jetEfficiencyBins),
    metEfficiencyBins=cms.vdouble(metEfficiencyBins),
    mhtEfficiencyBins=cms.vdouble(mhtEfficiencyBins),
    ettEfficiencyBins=cms.vdouble(ettEfficiencyBins),
    httEfficiencyBins=cms.vdouble(httEfficiencyBins),

    recoHTTMaxEta=cms.double(2.5),
    recoMHTMaxEta=cms.double(2.5),
)

l1tStage2CaloLayer2OfflineDQMEmu = l1tStage2CaloLayer2OfflineDQM.clone(
    stage2CaloLayer2JetSource=cms.InputTag("simCaloStage2Digis"),
    stage2CaloLayer2EtSumSource=cms.InputTag("simCaloStage2Digis"),

    histFolder=cms.string('L1TEMU/L1TStage2CaloLayer2'),
)
