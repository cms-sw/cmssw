import FWCore.ParameterSet.Config as cms

electronEfficiencyThresholds = [34, 36, 38, 40, 42]

electronEfficiencyBins = []
electronEfficiencyBins.extend(list(xrange(2, 42, 2)))
electronEfficiencyBins.extend(list(xrange(42, 45, 3)))
electronEfficiencyBins.extend(list(xrange(45, 50, 5)))
electronEfficiencyBins.extend(list(xrange(50, 70, 10)))
electronEfficiencyBins.extend(list(xrange(70, 101, 30)))

# additional efficiency vs eta, phi and # vertices plots will
# be created for the following probe electron pT thresholds
deepInspectionElectronThresholds = [48, 50]

# offset for 2D efficiency plots, uses
# electronEfficiencyBins + probeToL1Offset (GeV)
probeToL1Offset = 10

# just copy for now
photonEfficiencyThresholds = electronEfficiencyThresholds
photonEfficiencyBins = electronEfficiencyBins

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
    probeToL1Offset=cms.double(probeToL1Offset),
    deepInspectionElectronThresholds=cms.vdouble(deepInspectionElectronThresholds),

    photonEfficiencyThresholds=cms.vdouble(photonEfficiencyThresholds),
    photonEfficiencyBins=cms.vdouble(photonEfficiencyBins),
)

# modifications for the pp reference run
electronEfficiencyThresholds_HI = [5, 10, 15, 21]
deepInspectionElectronThresholds_HI = [15]

electronEfficiencyBins_HI = []
electronEfficiencyBins_HI.extend(list(xrange(1, 26, 1)))
electronEfficiencyBins_HI.extend(list(xrange(26, 42, 2)))
electronEfficiencyBins_HI.extend(list(xrange(42, 45, 3)))
electronEfficiencyBins_HI.extend(list(xrange(45, 50, 5)))
electronEfficiencyBins_HI.extend(list(xrange(50, 70, 10)))
electronEfficiencyBins_HI.extend(list(xrange(70, 101, 30)))

photonEfficiencyThresholds_HI = electronEfficiencyThresholds_HI
photonEfficiencyBins_HI = electronEfficiencyBins_HI

from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017
ppRef_2017.toModify(l1tEGammaOfflineDQM,
    TriggerFilter=cms.InputTag('hltEle20WPLoose1GsfTrackIsoFilter', '', 'HLT'),
    TriggerPath=cms.string('HLT_Ele20_WPLoose_Gsf_v4'),
    electronEfficiencyThresholds=cms.vdouble(electronEfficiencyThresholds_HI),
    electronEfficiencyBins=cms.vdouble(electronEfficiencyBins_HI),
    deepInspectionElectronThresholds=cms.vdouble(deepInspectionElectronThresholds_HI),
    photonEfficiencyThresholds=cms.vdouble(photonEfficiencyThresholds_HI),
    photonEfficiencyBins=cms.vdouble(photonEfficiencyBins_HI)
)

# emulator module
l1tEGammaOfflineDQMEmu = l1tEGammaOfflineDQM.clone(
    stage2CaloLayer2EGammaSource=cms.InputTag("simCaloStage2Digis"),

    histFolder=cms.string('L1TEMU/L1TEGamma'),
)

