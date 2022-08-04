import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.METMonitor_cfi import hltMETmonitoring

# HLT_PFMET110_PFMHT110_IDTight
PFMET110_PFMHT110_IDTight_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMET110/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET110_PFMHT110_IDTight_v*"]),
    enableFullMonitoring = True,
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
#HLT_PFMET120_PFMHT120_IDTight
PFMET120_PFMHT120_IDTight_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMET120/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET120_PFMHT120_IDTight_v*"]),
    enableFullMonitoring = True,
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFMET130_PFMHT130_IDTight
PFMET130_PFMHT130_IDTight_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMET130/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET130_PFMHT130_IDTight_v*"]),
    enableFullMonitoring = True,
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFMET140_PFMHT140_IDTight
PFMET140_PFMHT140_IDTight_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMET140/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET140_PFMHT140_IDTight_v*"]),
    enableFullMonitoring = True,
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFMETTypeOne110_PFMHT110_IDTight                                                                                                                              
PFMETTypeOne110_PFMHT110_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMETTypeOne110/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETTypeOne110_PFMHT110_IDTight_v*"]),
    enableFullMonitoring = True,
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFMETTypeOne120_PFMHT120_IDTight
PFMETTypeOne120_PFMHT120_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMETTypeOne120/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETTypeOne120_PFMHT120_IDTight_v*"]),
    enableFullMonitoring = True,
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFMETTypeOne130_PFMHT130_IDTight
PFMETTypeOne130_PFMHT130_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMETTypeOne130/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETTypeOne130_PFMHT130_IDTight_v*"]),
    enableFullMonitoring = True,
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFMETTypeOne140_PFMHT140_IDTight
PFMETTypeOne140_PFMHT140_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMETTypeOne140/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETTypeOne140_PFMHT140_IDTight_v*"]),
    enableFullMonitoring = True,
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFMETNoMu110_PFMHTNoMu110_IDTight
PFMETNoMu110_PFMHTNoMu110_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMETNoMu110/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_v*"]),
    enableFullMonitoring = True,
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFMETNoMu120_PFMHTNoMu120_IDTight
PFMETNoMu120_PFMHTNoMu120_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMETNoMu120/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"]),
    enableFullMonitoring = True,
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFMETNoMu130_PFMHTNoMu130_IDTight
PFMETNoMu130_PFMHTNoMu130_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMETNoMu130/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v*"]),
    enableFullMonitoring = True,
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFMETNoMu140_PFMHTNoMu140_IDTight
PFMETNoMu140_PFMHTNoMu140_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMETNoMu140/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v*"]),
    enableFullMonitoring = True,
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF
PFMETNoMu110_PFMHTNoMu110_FilterHF_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFMETNoMu110FilterHF',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF_v*"]),
    enableFullMonitoring = True,
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_FilterHF
PFMETNoMu120_PFMHTNoMu120_FilterHF_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFMETNoMu120FilterHF',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_FilterHF_v*"]),
    enableFullMonitoring = True,
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_FilterHF
PFMETNoMu130_PFMHTNoMu130_FilterHF_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFMETNoMu130FilterHF',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_FilterHF_v*"]),
    enableFullMonitoring = True,
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_FilterHF
PFMETNoMu140_PFMHTNoMu140_FilterHF_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFMETNoMu140FilterHF',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_FilterHF_v*"]),
    enableFullMonitoring = True,
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_IDTight
MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/MonoCentralPFJet80_PFMETNoMu110/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_IDTight"]),
    enableFullMonitoring = False,
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight
MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/MonoCentralPFJet80_PFMETNoMu120/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"]),
    enableFullMonitoring = False,
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_IDTight
MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/MonoCentralPFJet80_PFMETNoMu130/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_IDTight_v*"]),
    enableFullMonitoring = False,
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_IDTight
MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/MonoCentralPFJet80_PFMETNoMu140/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_IDTight_v*"]),
    enableFullMonitoring = False,
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFHT500_PFMET100_PFMHT100_IDTight
PFHT500_PFMET100_PFMHT100_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFHT500_PFMET100_PFMHT100/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT500_PFMET100_PFMHT100_IDTight_v*"]),
    enableFullMonitoring = False
)
# HLT_PFHT500_PFMET110_PFMHT110_IDTight
PFHT500_PFMET110_PFMHT110_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFHT500_PFMET110_PFMHT110/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT500_PFMET110_PFMHT110_IDTight_v*"]),
    enableFullMonitoring = False,
)
# HLT_PFHT700_PFMET85_PFMHT85_IDTight
PFHT700_PFMET85_PFMHT85_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFHT700_PFMET85_PFMHT85/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT700_PFMET85_PFMHT85_IDTight_v*"]),
    enableFullMonitoring = False,
)
# HLT_PFHT700_PFMET95_PFMHT95_IDTight
PFHT700_PFMET95_PFMHT95_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFHT700_PFMET95_PFMHT95/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT700_PFMET95_PFMHT95_IDTight_v*"]),
    enableFullMonitoring = False,
)
# HLT_PFHT800_PFMET75_PFMHT75_IDTight
PFHT800_PFMET75_PFMHT75_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFHT800_PFMET75_PFMHT75/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT800_PFMET75_PFMHT75_IDTight_v*"]),
    enableFullMonitoring = False,
)
# HLT_PFHT800_PFMET85_PFMHT85_IDTight
PFHT800_PFMET85_PFMHT85_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFHT800_PFMET85_PFMHT85/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT800_PFMET85_PFMHT85_IDTight_v*"]),
    enableFullMonitoring = False,
)
# HLT_PFMET120_PFMHT120_IDTight_PFHT60
PFMET120_PFMHT120_IDTight_PFHT60_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFMET120_PFHT60/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET120_PFMHT120_IDTight_PFHT60_v*"]),
    enableFullMonitoring = False,
    jetSelection      = "pt > 70 && abs(eta) < 2.4 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60
PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFMETNoMu120_PFHT60/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v*"]),
    enableFullMonitoring = False,
    jetSelection      = cms.string("pt > 70 && abs(eta) < 2.4 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")
)
# HLT_PFMETTypeOne120_PFMHT120_IDTight_PFHT60
PFMETTypeOne120_PFMHT120_IDTight_PFHT60_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFMETTypeOne120_PFMHT120_PFHT60/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETTypeOne120_PFMHT120_IDTight_PFHT60_v*"]),
    enableFullMonitoring = False,
    jetSelection      = "pt > 70 && abs(eta) < 2.4 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
## Add Pure MET Trigger ##
# HLT_L1ETMHadSeeds_v 
L1ETMHadSeeds_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/L1ETMHadSeeds/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_L1ETMHadSeeds_v*"]),
    enableFullMonitoring = True
)
# HLT_CaloMHT90_v 
CaloMHT90_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/CaloMHT/CaloMHT90/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_CaloMHT90_v*"]),
    enableFullMonitoring = False
)
# HLT_CaloMET80_NotCleaned_v 
CaloMET80_NotCleaned_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/Calo/CaloMET80_NotCleaned/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_CaloMET80_NotCleaned_v*"]),
    enableFullMonitoring = True
)
# HLT_CaloMET90_NotCleaned_v 
CaloMET90_NotCleaned_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/Calo/CaloMET90_NotCleaned/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_CaloMET90_NotCleaned_v*"]),
    enableFullMonitoring = True
)
# HLT_CaloMET100_NotCleaned_v 
CaloMET100_NotCleaned_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/Calo/CaloMET100_NotCleaned/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_CaloMET100_NotCleaned_v*"]),
    enableFullMonitoring = True
)
# HLT_CaloMET110_NotCleaned_v 
CaloMET110_NotCleaned_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/Calo/CaloMET110_NotCleaned/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_CaloMET110_NotCleaned_v*"]),
    enableFullMonitoring = True
)
# HLT_PFMET200_BeamHaloCleaned_v 
PFMET200_BeamHaloCleaned_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMET200_BeamHaloCleaned/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET200_BeamHaloCleaned_v*"]),
    enableFullMonitoring = True
)
# HLT_PFMET200_NotCleaned_v 
PFMET200_NotCleaned_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMET200_NotCleaned/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET200_NotCleaned_v*"]),
    enableFullMonitoring = True
)
# HLT_PFMET250_NotCleaned_v 
PFMET250_NotCleaned_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMET250_NotCleaned/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET250_NotCleaned_v*"]),
    enableFullMonitoring = True
)
# HLT_PFMET300_NotCleaned_v 
PFMET300_NotCleaned_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMET300_NotCleaned/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET300_NotCleaned_v*"]),
    enableFullMonitoring = True
)
# HLT_PFMETTypeOne200_BeamHaloCleaned_v 
PFMETTypeOne200_BeamHaloCleaned_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMETTypeOne200_BeamHaloCleaned/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETTypeOne200_BeamHaloCleaned_v*"]),
    enableFullMonitoring = True
)
exoHLTMETmonitoring = cms.Sequence(
    PFMET110_PFMHT110_IDTight_METmonitoring
    + PFMET120_PFMHT120_IDTight_METmonitoring
    + PFMET130_PFMHT130_IDTight_METmonitoring
    + PFMET140_PFMHT140_IDTight_METmonitoring
    + PFMETTypeOne110_PFMHT110_METmonitoring
    + PFMETTypeOne120_PFMHT120_METmonitoring
    + PFMETTypeOne130_PFMHT130_METmonitoring
    + PFMETTypeOne140_PFMHT140_METmonitoring
    + PFMETNoMu110_PFMHTNoMu110_METmonitoring
    + PFMETNoMu120_PFMHTNoMu120_METmonitoring
    + PFMETNoMu130_PFMHTNoMu130_METmonitoring
    + PFMETNoMu140_PFMHTNoMu140_METmonitoring
    + PFMETNoMu110_PFMHTNoMu110_FilterHF_METmonitoring
    + PFMETNoMu120_PFMHTNoMu120_FilterHF_METmonitoring
    + PFMETNoMu130_PFMHTNoMu130_FilterHF_METmonitoring
    + PFMETNoMu140_PFMHTNoMu140_FilterHF_METmonitoring
    + MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_METmonitoring
    + MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_METmonitoring
    + MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_METmonitoring
    + MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_METmonitoring
    + PFHT500_PFMET100_PFMHT100_METmonitoring
    + PFHT500_PFMET110_PFMHT110_METmonitoring
    + PFHT700_PFMET85_PFMHT85_METmonitoring
    + PFHT700_PFMET95_PFMHT95_METmonitoring
    + PFHT800_PFMET75_PFMHT75_METmonitoring
    + PFHT800_PFMET85_PFMHT85_METmonitoring
    + PFMET120_PFMHT120_IDTight_PFHT60_METmonitoring
    + PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_METmonitoring
    + PFMETTypeOne120_PFMHT120_IDTight_PFHT60_METmonitoring
    + L1ETMHadSeeds_METmonitoring
    + CaloMHT90_METmonitoring
    + CaloMET80_NotCleaned_METmonitoring
    + CaloMET90_NotCleaned_METmonitoring 
    + CaloMET100_NotCleaned_METmonitoring   
    + CaloMET110_NotCleaned_METmonitoring
    + PFMET200_BeamHaloCleaned_METmonitoring    
    + PFMET200_NotCleaned_METmonitoring
    + PFMET250_NotCleaned_METmonitoring
    + PFMET300_NotCleaned_METmonitoring
    + PFMETTypeOne200_BeamHaloCleaned_METmonitoring  
)
