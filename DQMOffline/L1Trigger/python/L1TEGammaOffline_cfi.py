from builtins import range
import FWCore.ParameterSet.Config as cms
from DQMOffline.L1Trigger.L1THistDefinitions_cff import histDefinitions

electronEfficiencyThresholds = [34, 36, 38, 40, 42]

electronEfficiencyBins = []
electronEfficiencyBins.extend(list(range(2, 42, 2)))
electronEfficiencyBins.extend(list(range(42, 45, 3)))
electronEfficiencyBins.extend(list(range(45, 50, 5)))
electronEfficiencyBins.extend(list(range(50, 70, 10)))
electronEfficiencyBins.extend(list(range(70, 101, 30)))

# additional efficiency vs eta, phi and # vertices plots will
# be created for the following probe electron pT thresholds
deepInspectionElectronThresholds = [48, 50]

# offset for 2D efficiency plots, uses
# electronEfficiencyBins + probeToL1Offset (GeV)
probeToL1Offset = 10

# just copy for now
photonEfficiencyThresholds = electronEfficiencyThresholds
photonEfficiencyBins = electronEfficiencyBins

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tEGammaOfflineDQM = DQMEDAnalyzer(
    "L1TEGammaOffline",
    electronCollection=cms.InputTag("gedGsfElectrons"),
    photonCollection=cms.InputTag("photons"),
    caloJetCollection=cms.InputTag("ak4CaloJets"),
    caloMETCollection=cms.InputTag("caloMet"),
    conversionsCollection=cms.InputTag("allConversions"),
    PVCollection=cms.InputTag("offlinePrimaryVerticesWithBS"),
    beamSpotCollection=cms.InputTag("offlineBeamSpot"),

    triggerInputTag=cms.InputTag('hltTriggerSummaryAOD', '', 'HLT'),
    triggerProcess=cms.string('HLT'),
    triggerResults=cms.InputTag('TriggerResults', '', 'HLT'),
    triggerNames = cms.vstring(
        'HLT_Ele32_WPTight_Gsf_v*',
    ),

    stage2CaloLayer2EGammaSource=cms.InputTag("caloStage2Digis", "EGamma"),

    histFolder=cms.string('L1T/L1TObjects/L1TEGamma/L1TriggerVsReco'),

    electronEfficiencyThresholds=cms.vdouble(electronEfficiencyThresholds),
    electronEfficiencyBins=cms.vdouble(electronEfficiencyBins),
    probeToL1Offset=cms.double(probeToL1Offset),
    deepInspectionElectronThresholds=cms.vdouble(deepInspectionElectronThresholds),

    photonEfficiencyThresholds=cms.vdouble(photonEfficiencyThresholds),
    photonEfficiencyBins=cms.vdouble(photonEfficiencyBins),
    maxDeltaRForL1Matching=cms.double(0.3),
    maxDeltaRForHLTMatching=cms.double(0.3),
    recoToL1TThresholdFactor=cms.double(1.25),

    histDefinitions=cms.PSet(
        nVertex=histDefinitions.nVertex.clone(),
        ETvsET=histDefinitions.ETvsET.clone(),
        PHIvsPHI=histDefinitions.PHIvsPHI.clone(),
    ),
)

# modifications for the pp reference run
electronEfficiencyThresholds_HI = [5, 10, 15, 21]
deepInspectionElectronThresholds_HI = [15]

electronEfficiencyBins_HI = []
electronEfficiencyBins_HI.extend(list(range(1, 26, 1)))
electronEfficiencyBins_HI.extend(list(range(26, 42, 2)))
electronEfficiencyBins_HI.extend(list(range(42, 45, 3)))
electronEfficiencyBins_HI.extend(list(range(45, 50, 5)))
electronEfficiencyBins_HI.extend(list(range(50, 70, 10)))
electronEfficiencyBins_HI.extend(list(range(70, 101, 30)))

photonEfficiencyThresholds_HI = electronEfficiencyThresholds_HI
photonEfficiencyBins_HI = electronEfficiencyBins_HI

from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017
ppRef_2017.toModify(
    l1tEGammaOfflineDQM,
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

    histFolder=cms.string('L1TEMU/L1TObjects/L1TEGamma/L1TriggerVsReco'),
)
