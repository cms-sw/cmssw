from builtins import range
import FWCore.ParameterSet.Config as cms
from DQMOffline.L1Trigger.L1THistDefinitions_cff import histDefinitions

jetEfficiencyThresholds = [36, 68, 128, 176]
metEfficiencyThresholds = [40, 60, 80, 100, 120]
mhtEfficiencyThresholds = [40, 60, 80, 100, 120]
ettEfficiencyThresholds = [30, 50, 90, 140]
httEfficiencyThresholds = [120, 160, 200, 240, 280]

jetEfficiencyBins = []
jetEfficiencyBins.extend(list(range(0, 120, 10)))
jetEfficiencyBins.extend(list(range(120, 180, 20)))
jetEfficiencyBins.extend(list(range(180, 300, 40)))
jetEfficiencyBins.extend(list(range(300, 401, 100)))

metEfficiencyBins = []
metEfficiencyBins.extend(list(range(0, 40, 4)))
metEfficiencyBins.extend(list(range(40, 70, 2)))
metEfficiencyBins.extend(list(range(70, 100, 5)))
metEfficiencyBins.extend(list(range(100, 160, 10)))
metEfficiencyBins.extend(list(range(160, 261, 20)))

mhtEfficiencyBins = []
mhtEfficiencyBins.extend(list(range(30, 50, 1)))
mhtEfficiencyBins.extend(list(range(50, 80, 5)))
mhtEfficiencyBins.extend(list(range(80, 140, 10)))
mhtEfficiencyBins.extend(list(range(140, 200, 15)))
mhtEfficiencyBins.extend(list(range(200, 300, 20)))
mhtEfficiencyBins.extend(list(range(300, 401, 50)))

ettEfficiencyBins = []
ettEfficiencyBins.extend(list(range(0, 30, 30)))
ettEfficiencyBins.extend(list(range(30, 50, 10)))
ettEfficiencyBins.extend(list(range(50, 90, 5)))
ettEfficiencyBins.extend(list(range(90, 141, 2)))

httEfficiencyBins = []
httEfficiencyBins.extend(list(range(0, 100, 5)))
httEfficiencyBins.extend(list(range(100, 200, 10)))
httEfficiencyBins.extend(list(range(200, 400, 20)))
httEfficiencyBins.extend(list(range(400, 500, 50)))
httEfficiencyBins.extend(list(range(500, 601, 10)))

# from https://twiki.cern.ch/twiki/bin/view/CMS/JetID13TeVRun2017
centralJetSelection = [
    'abs(eta) <= 2.7',
    'neutralHadronEnergyFraction < 0.9',
    'neutralEmEnergyFraction < 0.9',
    'numberOfDaughters > 1',
    'muonEnergyFraction < 0.8'
]
withinTrackerSelection = centralJetSelection[:]
withinTrackerSelection += [
    'abs(eta) <= 2.4',
    'chargedHadronEnergyFraction > 0',
    'chargedMultiplicity > 0',
    'chargedEmEnergyFraction < 0.8'
]
forwardJetSelection = [
    'abs(eta) > 2.7',
    'abs(eta) <= 3.0',
    'neutralEmEnergyFraction > 0.02',
    'neutralEmEnergyFraction < 0.99',
    'neutralMultiplicity > 2'
]
veryForwardJetSelection = [
    'abs(eta) > 3.0',
    'neutralEmEnergyFraction < 0.99',
    'neutralHadronEnergyFraction > 0.02',
    'neutralMultiplicity > 10'
]
centralJetSelection = ' && '.join(centralJetSelection)
withinTrackerSelection = ' && '.join(withinTrackerSelection)
forwardJetSelection = ' && '.join(forwardJetSelection)
veryForwardJetSelection = ' && '.join(veryForwardJetSelection)
completeSelection = 'et > 30 && (' + ' || '.join([centralJetSelection, withinTrackerSelection,
                                       forwardJetSelection, veryForwardJetSelection]) + ')'

goodPFJetsForL1T = cms.EDFilter(
    "PFJetSelector",
    src=cms.InputTag("ak4PFJetsPuppi"),
    cut=cms.string(completeSelection),
    filter=cms.bool(True),
)

from L1Trigger.L1TNtuples.L1TPFMetNoMuProducer_cfi import l1tPFMetNoMu

l1tPFMetNoMuForDQM = l1tPFMetNoMu.clone(
    pfMETCollection= 'pfMet',  ## Was 'pfMETT1', threw errors - AWB 2022.09.28
    muonCollection= 'muons'
)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tEtSumJetOfflineDQM = DQMEDAnalyzer(
    "L1TStage2CaloLayer2Offline",
    electronCollection=cms.InputTag("gedGsfElectrons"),
    pfJetCollection=cms.InputTag("goodPFJetsForL1T"),
    caloMETCollection=cms.InputTag("caloMetBE"),
    # MET collection including HF
    caloETMHFCollection=cms.InputTag("caloMet"),
    pfMETNoMuCollection=cms.InputTag('l1tPFMetNoMuForDQM'),
    conversionsCollection=cms.InputTag("allConversions"),
    PVCollection=cms.InputTag("offlinePrimaryVerticesWithBS"),
    beamSpotCollection=cms.InputTag("offlineBeamSpot"),

    triggerInputTag=cms.InputTag('hltTriggerSummaryAOD', '', 'HLT'),
    triggerProcess=cms.string('HLT'),
    triggerResults=cms.InputTag('TriggerResults', '', 'HLT'),
    triggerNames=cms.vstring(
        'HLT_IsoMu18_v*',
        'HLT_IsoMu20_v*',
        'HLT_IsoMu22_v*',
        'HLT_IsoMu24_v*',
        'HLT_IsoMu27_v*',
        'HLT_IsoMu30_v*',
    ),

    stage2CaloLayer2JetSource=cms.InputTag("caloStage2Digis", "Jet"),
    stage2CaloLayer2EtSumSource=cms.InputTag("caloStage2Digis", "EtSum"),

    histFolderEtSum=cms.string('L1T/L1TObjects/L1TEtSum/L1TriggerVsReco'),
    histFolderJet=cms.string('L1T/L1TObjects/L1TJet/L1TriggerVsReco'),
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

    histDefinitions=cms.PSet(
        nVertex=histDefinitions.nVertex.clone(),
        ETvsET=histDefinitions.ETvsET.clone(),
        PHIvsPHI=histDefinitions.PHIvsPHI.clone(),
        # L1JetETvsCaloJetET_HB=histDefinitions.ETvsET.clone(
        #     name='L1JetETvsCaloJetET_HB',
        #     title='L1 Jet E_{T} vs Offline Jet E_{T} (HB); Offline Jet E_{T} (GeV); L1 Jet E_{T} (GeV)',
        # )
    ),
)

# modifications for the pp reference run
jetEfficiencyThresholds_HI = [8, 16, 24, 44, 60, 80, 90]
jetEfficiencyBins_HI = []
jetEfficiencyBins_HI.extend(list(range(0, 60, 2)))
jetEfficiencyBins_HI.extend(list(range(60, 90, 5)))
jetEfficiencyBins_HI.extend(list(range(90, 120, 10)))
jetEfficiencyBins_HI.extend(list(range(120, 180, 20)))
jetEfficiencyBins_HI.extend(list(range(180, 300, 40)))
jetEfficiencyBins_HI.extend(list(range(300, 401, 100)))

from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017
ppRef_2017.toModify(
    l1tEtSumJetOfflineDQM,
    TriggerFilter=cms.InputTag('hltEle20WPLoose1GsfTrackIsoFilter', '', 'HLT'),
    TriggerPath=cms.string('HLT_Ele20_WPLoose_Gsf_v4'),
    jetEfficiencyThresholds=cms.vdouble(jetEfficiencyThresholds_HI),
    jetEfficiencyBins=cms.vdouble(jetEfficiencyBins_HI),
)

# emulator module
l1tEtSumJetOfflineDQMEmu = l1tEtSumJetOfflineDQM.clone(
    stage2CaloLayer2JetSource = "simCaloStage2Digis",
    stage2CaloLayer2EtSumSource = "simCaloStage2Digis",

    histFolderEtSum = 'L1TEMU/L1TObjects/L1TEtSum/L1TriggerVsReco',
    histFolderJet= 'L1TEMU/L1TObjects/L1TJet/L1TriggerVsReco'
)

# sequences
l1tEtSumJetOfflineDQMSeq = cms.Sequence(
    cms.ignore(goodPFJetsForL1T)
    + l1tPFMetNoMuForDQM
    + l1tEtSumJetOfflineDQM
)

l1tEtSumJetOfflineDQMEmuSeq = cms.Sequence(
    cms.ignore(goodPFJetsForL1T)
    + l1tPFMetNoMuForDQM
    + l1tEtSumJetOfflineDQMEmu
)

