
import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.HTMonitor_cfi import hltHTmonitoring

# HLT_PFMETNoMu120_PFMHTNoMu120_IDTight
PFMETNoMu120_PFMHTNoMu120_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMETNoMu120/',
    enableFullMonitoring = False,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"])
)

# HLT_MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_IDTight
MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/MET/MonoCentralPFJet80_PFMETNoMu110/',
    jetSelection      = "pt > 100 && eta < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1",
    jetSelection_HT   = "pt > 30 && eta < 2.5",
    enableFullMonitoring = False,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_IDTight_v*"])
)
# HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight
MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/MET/MonoCentralPFJet80_PFMETNoMu120/',
    jetSelection      = "pt > 100 && eta < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1",
    jetSelection_HT   = "pt > 30 && eta < 2.5",
    enableFullMonitoring = False,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"])
)
# HLT_MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_IDTight
MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/MET/MonoCentralPFJet80_PFMETNoMu130/',
    jetSelection      = "pt > 100 && eta < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1",
    jetSelection_HT   = "pt > 30 && eta < 2.5",
    enableFullMonitoring = False,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_IDTight_v*"])
)
# HLT_MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_IDTight
MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/MET/MonoCentralPFJet80_PFMETNoMu140/',
    jetSelection      = "pt > 100 && eta < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1",
    jetSelection_HT   = "pt > 30 && eta < 2.5",
    enableFullMonitoring = False,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_IDTight_v*"])
)
# HLT_PFHT350MinPFJet15
PFHT350MinPFJet15_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/HT/HLT_PFHT350MinPFJet15/',
    jetSelection      = "pt > 15",
    jetSelection_HT   = "pt > 30 && eta < 2.5",
    enableFullMonitoring = False,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT350MinPFJet15_v*"])
)
# HLT_PFHT500_PFMET100_PFMHT100_IDTight                                                                                                                                                          
PFHT500_PFMET100_PFMHT100_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/HT/PFHT500_PFMET100_PFMHT100/',
    metSelection      = "pt > 200",
    jetSelection      = "pt > 0",
    jetSelection_HT   = "pt > 30 && eta < 2.5",
    enableFullMonitoring = False,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT500_PFMET100_PFMHT100_IDTight_v*"])
)
# HLT_PFHT500_PFMET110_PFMHT110_IDTight                                                                                                                                                         
PFHT500_PFMET110_PFMHT110_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/HT/PFHT500_PFMET110_PFMHT110/',
    metSelection      = "pt > 210",
    jetSelection      = "pt > 0",
    jetSelection_HT   = "pt > 30 && eta < 2.5",
    enableFullMonitoring = False,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT500_PFMET110_PFMHT110_IDTight_v*"])
)
# HLT_PFHT700_PFMET85_PFMHT85_IDTight                                                                                                                                                            
PFHT700_PFMET85_PFMHT85_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/HT/PFHT700_PFMET85_PFMHT85/',
    metSelection      = "pt > 185",
    jetSelection      = "pt > 0",
    jetSelection_HT   = "pt > 30 && eta < 2.5",
    enableFullMonitoring = False,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT700_PFMET85_PFMHT85_IDTight_v*"])
)
# HLT_PFHT700_PFMET95_PFMHT95_IDTight                                                                                                                                                            
PFHT700_PFMET95_PFMHT95_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/HT/PFHT700_PFMET95_PFMHT95/',
    metSelection      = "pt > 195",
    jetSelection      = "pt > 0",
    jetSelection_HT   = "pt > 30 && eta < 2.5",
    enableFullMonitoring = False,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT700_PFMET95_PFMHT95_IDTight_v*"])
)
# HLT_PFHT800_PFMET75_PFMHT75_IDTight                                                                                                                                                            
PFHT800_PFMET75_PFMHT75_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/HT/PFHT800_PFMET75_PFMHT75/',
    metSelection      = "pt > 175",
    jetSelection      = "pt > 0",
    jetSelection_HT   = "pt > 30 && eta < 2.5",
    enableFullMonitoring = False,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT800_PFMET75_PFMHT75_IDTight_v*"])
)
# HLT_PFHT800_PFMET85_PFMHT85_IDTight                                                                                                                                                          
PFHT800_PFMET85_PFMHT85_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/HT/PFHT800_PFMET85_PFMHT85/',
    metSelection      = "pt > 185",
    jetSelection      = "pt > 0",
    jetSelection_HT   = "pt > 30 && eta < 2.5",
    enableFullMonitoring = False,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT800_PFMET85_PFMHT85_IDTight_v*"])
)
# HLT_PFHT1050_v7
PFHT1050_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/JME/HT/PFHT1050/',
    jetSelection      = "pt > 0",
    jetSelection_HT   = "pt > 30 && eta < 2.5",
    enableFullMonitoring = True,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT1050_v*"])
)
#HLT_PFHT890
PFHT890_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/JME/HT/PFHT890/',
    jetSelection_HT   = "pt > 30 && eta < 2.5",
    enableFullMonitoring = True,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT890_v*"])
)
#HLT_PFHT780
PFHT780_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/JME/HT/PFHT780/',
    jetSelection_HT   = "pt > 30 && eta < 2.5",
    enableFullMonitoring = True,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT780_v*"])
)
#HLT_PFHT680
PFHT680_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/JME/HT/PFHT680/',
    jetSelection_HT   = "pt > 30 && eta < 2.5",
    enableFullMonitoring = True,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT680_v*"])

)
#HLT_PFHT590
PFHT590_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/JME/HT/PFHT590/',
    jetSelection_HT   = "pt > 30 && eta < 2.5",
    enableFullMonitoring = True,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT590_v*"])
)
#HLT_PFHT510
PFHT510_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/JME/HT/PFHT510/',
    jetSelection_HT   = "pt > 30 && eta < 2.5",
    enableFullMonitoring = True,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT510_v*"])
)
#HLT_PFHT430
PFHT430_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/JME/HT/PFHT430/',
    jetSelection_HT   = "pt > 30 && eta < 2.5",
    enableFullMonitoring = True,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT430_v*"])
)
#HLT_PFHT370
PFHT370_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/JME/HT/PFHT370/',
    jetSelection_HT   = "pt > 30 && eta < 2.5",
    enableFullMonitoring = True,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT370_v*"])
)
#HLT_PFHT250
PFHT250_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/JME/HT/PFHT250/',
    jetSelection_HT   = "pt > 30 && eta < 2.5",
    enableFullMonitoring = True,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT250_v*"])
)
#HLT_PFHT180
PFHT180_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/JME/HT/PFHT180/',
    jetSelection_HT   = "pt > 30 && eta < 2.5",
    enableFullMonitoring = True,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT180_v*"])
)
# HLT_PFMETTypeOne110_PFMHT110_IDTight                                                                                                                              
PFMETTypeOne110_PFMHT110_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFMETTypeOne110/',
    jetSelection      = "pt > 100 && eta < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1",
    jetSelection_HT   = "pt > 30 && eta < 2.5",
    enableFullMonitoring = False,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETTypeOne110_PFMHT110_IDTight_v*"])
)
# HLT_PFMETTypeOne120_PFMHT120_IDTight
PFMETTypeOne120_PFMHT120_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFMETTypeOne120/',
    jetSelection      = "pt > 100 && eta < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1",
    jetSelection_HT   = "pt > 30 && eta < 2.5",
    enableFullMonitoring = False,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PPFMETTypeOne120_PFMHT120_IDTight_v*"])
)
# HLT_PFMETTypeOne130_PFMHT130_IDTight
PFMETTypeOne130_PFMHT130_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFMETTypeOne130/',
    jetSelection      = "pt > 100 && eta < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1",
    jetSelection_HT   = "pt > 30 && eta < 2.5",
    enableFullMonitoring = False,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETTypeOne130_PFMHT130_IDTight_v*"])
)
# HLT_PFMETTypeOne140_PFMHT140_IDTight
PFMETTypeOne140_PFMHT140_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFMETTypeOne140/',
    jetSelection      = "pt > 100 && eta < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1",
    jetSelection_HT   = "pt > 30 && eta < 2.5",
    enableFullMonitoring = False,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETTypeOne140_PFMHT140_IDTight_v*"])
)
# HLT_PFMET120_PFMHT120_IDTight_PFHT60
PFMET120_PFMHT120_IDTight_PFHT60_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFMET120_PFHT60/',
    metSelection      = "pt > 220",
    jetSelection      = "pt > 70 && abs(eta) < 2.4 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1",
    enableFullMonitoring = False,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMET120_PFMHT120_IDTight_PFHT60_v*"])
)
# HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60
PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFMETNoMu120_PFHT60/',
    metSelection      = "pt > 220",
    jetSelection      = "pt > 70 && abs(eta) < 2.4 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1",
    enableFullMonitoring = False,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v*"])
)
# HLT_PFMETTypeOne120_PFMHT120_IDTight_PFHT60
PFMETTypeOne120_PFMHT120_IDTight_PFHT60_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/EXO/MET/PFMETTypeOne120_PFHT60/',
    metSelection      = "pt > 220",
    jetSelection      = "pt > 70 && abs(eta) < 2.4 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1",
    enableFullMonitoring = False,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFMETTypeOne120_PFMHT120_IDTight_PFHT60_v*"])
)
exoHLTHTmonitoring = cms.Sequence(
    PFMETNoMu120_PFMHTNoMu120_HTmonitoring
    + MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_HTmonitoring
    + PFHT350MinPFJet15_HTmonitoring
    + PFHT500_PFMET100_PFMHT100_HTmonitoring
    + PFHT500_PFMET110_PFMHT110_HTmonitoring
    + PFHT700_PFMET85_PFMHT85_HTmonitoring
    + PFHT700_PFMET95_PFMHT95_HTmonitoring
    + PFHT800_PFMET75_PFMHT75_HTmonitoring
    + PFHT800_PFMET85_PFMHT85_HTmonitoring
    + PFHT1050_HTmonitoring
    + PFHT890_HTmonitoring
    + PFHT780_HTmonitoring
    + PFHT680_HTmonitoring
    + PFHT590_HTmonitoring
    + PFHT510_HTmonitoring
    + PFHT430_HTmonitoring
    + PFHT370_HTmonitoring
    + PFHT250_HTmonitoring
    + PFHT180_HTmonitoring
    + PFMETTypeOne110_PFMHT110_HTmonitoring
    + PFMETTypeOne120_PFMHT120_HTmonitoring
    + PFMETTypeOne130_PFMHT130_HTmonitoring
    + PFMETTypeOne140_PFMHT140_HTmonitoring
    + PFMET120_PFMHT120_IDTight_PFHT60_HTmonitoring
    + PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_HTmonitoring
    + PFMETTypeOne120_PFMHT120_IDTight_PFHT60_HTmonitoring
)

