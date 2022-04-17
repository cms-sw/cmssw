import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.METMonitor_cfi import hltMETmonitoring

# HLT_PFMET110_PFMHT110_IDTight
PFMET110_PFMHT110_IDTight_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMET110/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET110_PFMHT110_IDTight_v*"]),
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
#HLT_PFMET120_PFMHT120_IDTight
PFMET120_PFMHT120_IDTight_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMET120',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET120_PFMHT120_IDTight_v*"]),
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFMET130_PFMHT130_IDTight
PFMET130_PFMHT130_IDTight_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMET130/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET130_PFMHT130_IDTight_v*"]),
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFMET140_PFMHT140_IDTight
PFMET140_PFMHT140_IDTight_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMET140/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET140_PFMHT140_IDTight_v*"]),
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFMETTypeOne110_PFMHT110_IDTight                                                                                                                              
PFMETTypeOne110_PFMHT110_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMETTypeOne110/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETTypeOne110_PFMHT110_IDTight_v*"]),
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFMETTypeOne120_PFMHT120_IDTight
PFMETTypeOne120_PFMHT120_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMETTypeOne120/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PPFMETTypeOne120_PFMHT120_IDTight_v*"]),
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFMETTypeOne130_PFMHT130_IDTight
PFMETTypeOne130_PFMHT130_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMETTypeOne130/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETTypeOne130_PFMHT130_IDTight_v*"]),
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFMETTypeOne140_PFMHT140_IDTight
PFMETTypeOne140_PFMHT140_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMETTypeOne140/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETTypeOne140_PFMHT140_IDTight_v*"]),
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFMETNoMu90_PFMHTNoMu90_IDTight
PFMETNoMu90_PFMHTNoMu90_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMETNoMu90/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETNoMu90_PFMHTNoMu90_IDTight_v*"]),
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFMETNoMu110_PFMHTNoMu110_IDTight
PFMETNoMu110_PFMHTNoMu110_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMETNoMu110/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_v*"]),
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFMETNoMu120_PFMHTNoMu120_IDTight
PFMETNoMu120_PFMHTNoMu120_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMETNoMu120/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"]),
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFMETNoMu130_PFMHTNoMu130_IDTight
PFMETNoMu130_PFMHTNoMu130_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMETNoMu130/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v*"]),
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFMETNoMu140_PFMHTNoMu140_IDTight
PFMETNoMu140_PFMHTNoMu140_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMETNoMu140/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v*"]),
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_MET200
MET200_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/MET200/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_MET200_v*"])
)
# HLT_MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_IDTight
MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/MonoCentralPFJet80_PFMETNoMu110/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_IDTight"]),
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight
MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/MonoCentralPFJet80_PFMETNoMu120/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"]),
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_IDTight
MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/MonoCentralPFJet80_PFMETNoMu130/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_IDTight_v*"]),
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_IDTight
MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/MonoCentralPFJet80_PFMETNoMu140/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_IDTight_v*"]),
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_IDTight
MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/MonoCentralPFJet80_PFMETNoMu90/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_IDTight_v*"]),
    jetSelection      = "pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFHT500_PFMET100_PFMHT100_IDTight
PFHT500_PFMET100_PFMHT100_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFHT500_PFMET100_PFMHT100/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT500_PFMET100_PFMHT100_IDTight_v*"])
)
# HLT_PFHT500_PFMET110_PFMHT110_IDTight
PFHT500_PFMET110_PFMHT110_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFHT500_PFMET110_PFMHT110/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT500_PFMET110_PFMHT110_IDTight_v*"])
)
# HLT_PFHT700_PFMET85_PFMHT85_IDTight
PFHT700_PFMET85_PFMHT85_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFHT700_PFMET85_PFMHT85/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT700_PFMET85_PFMHT85_IDTight_v*"])
)
# HLT_PFHT700_PFMET95_PFMHT95_IDTight
PFHT700_PFMET95_PFMHT95_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFHT700_PFMET95_PFMHT95/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT700_PFMET95_PFMHT95_IDTight_v*"])
)
# HLT_PFHT800_PFMET75_PFMHT75_IDTight
PFHT800_PFMET75_PFMHT75_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFHT800_PFMET75_PFMHT75/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT800_PFMET75_PFMHT75_IDTight_v*"])
)
# HLT_PFHT800_PFMET85_PFMHT85_IDTight
PFHT800_PFMET85_PFMHT85_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFHT800_PFMET85_PFMHT85/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT800_PFMET85_PFMHT85_IDTight_v*"])
)
# HLT_PFMET120_PFMHT120_IDTight_PFHT60
PFMET120_PFMHT120_IDTight_PFHT60_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFMET120_PFHT60/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET120_PFMHT120_IDTight_PFHT60_v*"]),
    jetSelection      = "pt > 70 && abs(eta) < 2.4 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
# HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60
PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFMETNoMu120_PFHT60/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v*"]),
    jetSelection      = cms.string("pt > 70 && abs(eta) < 2.4 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")
)
# HLT_PFMETTypeOne120_PFMHT120_IDTight_PFHT60
PFMETTypeOne120_PFMHT120_IDTight_PFHT60_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFMETTypeOne120_PFHT60/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETTypeOne120_PFMHT120_IDTight_PFHT60_v*"]),
    jetSelection      = "pt > 70 && abs(eta) < 2.4 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1"
)
## Add Pure MET Trigger ##
# HLT_L1ETMHadSeeds_v 
L1ETMHadSeeds_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/L1ETMHadSeeds/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_L1ETMHadSeeds_v*"])
)
# HLT_CaloMHT90_v 
CaloMHT90_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/CaloMHT/CaloMHT90/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_CaloMHT90_v*"])
)
# HLT_CaloMET70_HBHECleaned_v 
CaloMET70_HBHECleaned_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/Calo/CaloMET70_HBHECleaned/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_CaloMET70_HBHECleaned_v*"])
)
# HLT_CaloMET80_HBHECleaned_v 
CaloMET80_HBHECleaned_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/Calo/CaloMET80_HBHECleaned/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_CaloMET80_HBHECleaned_v*"])
)
# HLT_CaloMET80_NotCleaned_v 
CaloMET80_NotCleaned_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/Calo/CaloMET80_NotCleaned/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_CaloMET80_NotCleaned_v*"])
)
# HLT_CaloMET90_HBHECleaned_v 
CaloMET90_HBHECleaned_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/Calo/CaloMET90_HBHECleaned/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_CaloMET90_HBHECleaned_v*"])
)
# HLT_CaloMET90_NotCleaned_v 
CaloMET90_NotCleaned_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/Calo/CaloMET90_NotCleaned/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_CaloMET90_NotCleaned_v*"])
)
# HLT_CaloMET100_HBHECleaned_v 
CaloMET100_HBHECleaned_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/Calo/CaloMET100_HBHECleaned/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_CaloMET100_HBHECleaned_v*"])
)
# HLT_CaloMET100_NotCleaned_v 
CaloMET100_NotCleaned_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/Calo/CaloMET100_NotCleaned/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_CaloMET100_NotCleaned_v*"])
)
# HLT_CaloMET110_NotCleaned_v 
CaloMET110_NotCleaned_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/Calo/CaloMET110_NotCleaned/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_CaloMET110_NotCleaned_v*"])
)
# HLT_CaloMET250_HBHECleaned_v 
CaloMET250_HBHECleaned_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/Calo/CaloMET250_HBHECleaned/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_CaloMET250_HBHECleaned_v*"])
)
# HLT_CaloMET250_NotCleaned_v 
CaloMET250_NotCleaned_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/Calo/CaloMET250_NotCleaned/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_CaloMET250_NotCleaned_v*"])
)
# HLT_CaloMET300_HBHECleaned_v 
CaloMET300_HBHECleaned_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/Calo/CaloMET300_HBHECleaned/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_CaloMET300_HBHECleaned_v*"])
)
# HLT_CaloMET300_HBHECleaned_v 
CaloMET350_HBHECleaned_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/Calo/CaloMET350_HBHECleaned/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_CaloMET350_HBHECleaned_v*"])
)
# HLT_CaloMET300_HBHECleaned_v 
PFMET200_HBHE_BeamHaloCleaned_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMET200_HBHE_BeamHaloCleaned/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET200_HBHE_BeamHaloCleaned_v*"])
)
# HLT_CaloMET300_HBHECleaned_v 
PFMET200_HBHECleaned_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMET200_HBHECleaned/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET200_HBHECleaned_v*"])
)
# HLT_CaloMET300_HBHECleaned_v 
PFMET200_NotCleaned_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMET200_NotCleaned/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET200_NotCleaned_v*"])
)
# HLT_CaloMET300_HBHECleaned_v 
PFMET250_HBHECleaned_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMET250_HBHECleaned/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET250_HBHECleaned_v*"])
)
# HLT_CaloMET300_HBHECleaned_v 
PFMET300_HBHECleaned_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMET300_HBHECleaned/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET300_HBHECleaned_v*"])
)
# HLT_CaloMET300_HBHECleaned_v 
PFMETTypeOne200_HBHE_BeamHaloCleaned_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMETTypeOne200_HBHE_BeamHaloCleaned/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETTypeOne200_HBHE_BeamHaloCleaned_v*"])
)
# HLT_L1MET_DTClusterNoMB1S50_v
L1MET_DTClusterNoMB1S50_METmonitoring = hltMETmonitoring.clone(
    FolderName = 'HLT/EXO/MET/L1MET_DTClusterNoMB1S50/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_L1MET_DTClusterNoMB1S50_v*"])
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
    + PFMETNoMu90_PFMHTNoMu90_METmonitoring
    + MET200_METmonitoring
    + MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_METmonitoring
    + PFMET120_PFMHT120_IDTight_PFHT60_METmonitoring
    + PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_METmonitoring
    + PFMETTypeOne120_PFMHT120_IDTight_PFHT60_METmonitoring
    + L1ETMHadSeeds_METmonitoring
    + CaloMHT90_METmonitoring
    + CaloMET70_HBHECleaned_METmonitoring
    + CaloMET80_HBHECleaned_METmonitoring
    + CaloMET80_NotCleaned_METmonitoring
    + CaloMET90_HBHECleaned_METmonitoring
    + CaloMET90_NotCleaned_METmonitoring 
    + CaloMET100_HBHECleaned_METmonitoring
    + CaloMET100_NotCleaned_METmonitoring   
    + CaloMET110_NotCleaned_METmonitoring
    + CaloMET250_HBHECleaned_METmonitoring
    + CaloMET250_NotCleaned_METmonitoring 
    + CaloMET300_HBHECleaned_METmonitoring 
    + CaloMET350_HBHECleaned_METmonitoring
    + PFMET200_HBHE_BeamHaloCleaned_METmonitoring  
    + PFMET200_HBHECleaned_METmonitoring  
    + PFMET200_NotCleaned_METmonitoring
    + PFMET250_HBHECleaned_METmonitoring
    + PFMET300_HBHECleaned_METmonitoring
    + PFMETTypeOne200_HBHE_BeamHaloCleaned_METmonitoring  
    + L1MET_DTClusterNoMB1S50_METmonitoring
)
