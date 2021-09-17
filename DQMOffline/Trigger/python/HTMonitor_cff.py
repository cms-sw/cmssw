
import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.HTMonitor_cfi import hltHTmonitoring

# HLT_PFMETNoMu90_PFMHTNoMu90_IDTight
PFMETNoMu90_PFMHTNoMu90_HTmonitoring = hltHTmonitoring.clone()
PFMETNoMu90_PFMHTNoMu90_HTmonitoring.FolderName = cms.string('HLT/JME/MET/PFMETNoMu90/')
PFMETNoMu90_PFMHTNoMu90_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMETNoMu90_PFMHTNoMu90_IDTight_v*")

# HLT_PFMETNoMu120_PFMHTNoMu120_IDTight
PFMETNoMu120_PFMHTNoMu120_HTmonitoring = hltHTmonitoring.clone()
PFMETNoMu120_PFMHTNoMu120_HTmonitoring.FolderName = cms.string('HLT/JME/MET/PFMETNoMu120/')
PFMETNoMu120_PFMHTNoMu120_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")

# HLT_MET200
MET200_HTmonitoring = hltHTmonitoring.clone()
MET200_HTmonitoring.FolderName = cms.string('HLT/JME/MET/MET200/')
MET200_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_MET200_v*")

# HLT_MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_IDTight
MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_HTmonitoring = hltHTmonitoring.clone()
MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_HTmonitoring.FolderName = cms.string('HLT/EXO/MET/MonoCentralPFJet80_PFMETNoMu110/')
MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_IDTight_v*")
MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_HTmonitoring.jetSelection      = cms.string("pt > 100 && eta < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")
MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_HTmonitoring.jetSelection_HT   = cms.string("pt > 30 && eta < 2.5")

# HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight
MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_HTmonitoring = hltHTmonitoring.clone()
MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_HTmonitoring.FolderName = cms.string('HLT/EXO/MET/MonoCentralPFJet80_PFMETNoMu120/')
MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")
MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_HTmonitoring.jetSelection      = cms.string("pt > 100 && eta < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")
MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_HTmonitoring.jetSelection_HT   = cms.string("pt > 30 && eta < 2.5")

# HLT_MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_IDTight
MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_HTmonitoring = hltHTmonitoring.clone()
MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_HTmonitoring.FolderName = cms.string('HLT/EXO/MET/MonoCentralPFJet80_PFMETNoMu130/')
MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_IDTight_v*")
MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_HTmonitoring.jetSelection      = cms.string("pt > 100 && eta < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")
MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_HTmonitoring.jetSelection_HT   = cms.string("pt > 30 && eta < 2.5")

# HLT_MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_IDTight
MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_HTmonitoring = hltHTmonitoring.clone()
MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_HTmonitoring.FolderName = cms.string('HLT/EXO/MET/MonoCentralPFJet80_PFMETNoMu140/')
MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_IDTight_v*")
MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_HTmonitoring.jetSelection      = cms.string("pt > 100 && eta < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")
MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_HTmonitoring.jetSelection_HT   = cms.string("pt > 30 && eta < 2.5")

# HLT_MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_IDTight
MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_HTmonitoring = hltHTmonitoring.clone()
MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_HTmonitoring.FolderName = cms.string('HLT/EXO/MET/MonoCentralPFJet80_PFMETNoMu90/')
MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_IDTight_v*")
MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_HTmonitoring.jetSelection_HT   = cms.string("pt > 30 && eta < 2.5")

# HLT_PFHT350MinPFJet15
PFHT350MinPFJet15_HTmonitoring = hltHTmonitoring.clone()
PFHT350MinPFJet15_HTmonitoring.FolderName = cms.string('HLT/HT/HLT_PFHT350MinPFJet15/')
PFHT350MinPFJet15_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT350MinPFJet15_v*")
PFHT350MinPFJet15_HTmonitoring.jetSelection      = cms.string("pt > 15")
PFHT350MinPFJet15_HTmonitoring.jetSelection_HT   = cms.string("pt > 30 && eta < 2.5")

# HLT_PFHT500_PFMET100_PFMHT100_IDTight                                                                                                                                                          
PFHT500_PFMET100_PFMHT100_HTmonitoring = hltHTmonitoring.clone()
PFHT500_PFMET100_PFMHT100_HTmonitoring.FolderName = cms.string('HLT/HT/PFHT500_PFMET100_PFMHT100/')
PFHT500_PFMET100_PFMHT100_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT500_PFMET100_PFMHT100_IDTight_v*")
PFHT500_PFMET100_PFMHT100_HTmonitoring.metSelection      = cms.string("pt > 200")
PFHT500_PFMET100_PFMHT100_HTmonitoring.jetSelection      = cms.string("pt > 0")
PFHT500_PFMET100_PFMHT100_HTmonitoring.jetSelection_HT   = cms.string("pt > 30 && eta < 2.5")

# HLT_PFHT500_PFMET110_PFMHT110_IDTight                                                                                                                                                         
PFHT500_PFMET110_PFMHT110_HTmonitoring = hltHTmonitoring.clone()
PFHT500_PFMET110_PFMHT110_HTmonitoring.FolderName = cms.string('HLT/HT/PFHT500_PFMET110_PFMHT110/')
PFHT500_PFMET110_PFMHT110_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT500_PFMET110_PFMHT110_IDTight_v*")
PFHT500_PFMET110_PFMHT110_HTmonitoring.metSelection      = cms.string("pt > 210")
PFHT500_PFMET110_PFMHT110_HTmonitoring.jetSelection      = cms.string("pt > 0")
PFHT500_PFMET110_PFMHT110_HTmonitoring.jetSelection_HT   = cms.string("pt > 30 && eta < 2.5")

# HLT_PFHT700_PFMET85_PFMHT85_IDTight                                                                                                                                                            
PFHT700_PFMET85_PFMHT85_HTmonitoring = hltHTmonitoring.clone()
PFHT700_PFMET85_PFMHT85_HTmonitoring.FolderName = cms.string('HLT/HT/PFHT700_PFMET85_PFMHT85/')
PFHT700_PFMET85_PFMHT85_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT700_PFMET85_PFMHT85_IDTight_v*")
PFHT700_PFMET85_PFMHT85_HTmonitoring.metSelection      = cms.string("pt > 185")
PFHT700_PFMET85_PFMHT85_HTmonitoring.jetSelection      = cms.string("pt > 0")
PFHT700_PFMET85_PFMHT85_HTmonitoring.jetSelection_HT   = cms.string("pt > 30 && eta < 2.5")

# HLT_PFHT700_PFMET95_PFMHT95_IDTight                                                                                                                                                            
PFHT700_PFMET95_PFMHT95_HTmonitoring = hltHTmonitoring.clone()
PFHT700_PFMET95_PFMHT95_HTmonitoring.FolderName = cms.string('HLT/HT/PFHT700_PFMET95_PFMHT95/')
PFHT700_PFMET95_PFMHT95_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT700_PFMET95_PFMHT95_IDTight_v*")
PFHT700_PFMET95_PFMHT95_HTmonitoring.metSelection      = cms.string("pt > 195")
PFHT700_PFMET95_PFMHT95_HTmonitoring.jetSelection      = cms.string("pt > 0")
PFHT700_PFMET95_PFMHT95_HTmonitoring.jetSelection_HT   = cms.string("pt > 30 && eta < 2.5")

# HLT_PFHT800_PFMET75_PFMHT75_IDTight                                                                                                                                                            
PFHT800_PFMET75_PFMHT75_HTmonitoring = hltHTmonitoring.clone()
PFHT800_PFMET75_PFMHT75_HTmonitoring.FolderName = cms.string('HLT/HT/PFHT800_PFMET75_PFMHT75/')
PFHT800_PFMET75_PFMHT75_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT800_PFMET75_PFMHT75_IDTight_v*")
PFHT800_PFMET75_PFMHT75_HTmonitoring.metSelection      = cms.string("pt > 175")
PFHT800_PFMET75_PFMHT75_HTmonitoring.jetSelection      = cms.string("pt > 0")
PFHT800_PFMET75_PFMHT75_HTmonitoring.jetSelection_HT   = cms.string("pt > 30 && eta < 2.5")

# HLT_PFHT800_PFMET85_PFMHT85_IDTight                                                                                                                                                          
PFHT800_PFMET85_PFMHT85_HTmonitoring = hltHTmonitoring.clone()
PFHT800_PFMET85_PFMHT85_HTmonitoring.FolderName = cms.string('HLT/HT/PFHT800_PFMET85_PFMHT85/')
PFHT800_PFMET85_PFMHT85_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT800_PFMET85_PFMHT85_IDTight_v*")
PFHT800_PFMET85_PFMHT85_HTmonitoring.metSelection      = cms.string("pt > 185")
PFHT800_PFMET85_PFMHT85_HTmonitoring.jetSelection      = cms.string("pt > 0")
PFHT800_PFMET85_PFMHT85_HTmonitoring.jetSelection_HT   = cms.string("pt > 30 && eta < 2.5")

# HLT_PFHT1050_v7
PFHT1050_HTmonitoring = hltHTmonitoring.clone()
PFHT1050_HTmonitoring.FolderName = cms.string('HLT/JME/HT/PFHT1050/')
PFHT1050_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT1050_v*")
PFHT1050_HTmonitoring.jetSelection      = cms.string("pt > 0")
PFHT1050_HTmonitoring.jetSelection_HT   = cms.string("pt > 30 && eta < 2.5")

#HLT_PFHT890
PFHT890_HTmonitoring = hltHTmonitoring.clone()
PFHT890_HTmonitoring.FolderName = cms.string('HLT/JME/HT/PFHT890/')
PFHT890_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT890_v*")
PFHT890_HTmonitoring.jetSelection_HT   = cms.string("pt > 30 && eta < 2.5")

#HLT_PFHT780
PFHT780_HTmonitoring = hltHTmonitoring.clone()
PFHT780_HTmonitoring.FolderName = cms.string('HLT/JME/HT/PFHT780/')
PFHT780_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT780_v*")
PFHT780_HTmonitoring.jetSelection_HT   = cms.string("pt > 30 && eta < 2.5")

#HLT_PFHT680
PFHT680_HTmonitoring = hltHTmonitoring.clone()
PFHT680_HTmonitoring.FolderName = cms.string('HLT/JME/HT/PFHT680/')
PFHT680_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT680_v*")
PFHT680_HTmonitoring.jetSelection_HT   = cms.string("pt > 30 && eta < 2.5")

#HLT_PFHT590
PFHT590_HTmonitoring = hltHTmonitoring.clone()
PFHT590_HTmonitoring.FolderName = cms.string('HLT/JME/HT/PFHT590/')
PFHT590_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT590_v*")
PFHT590_HTmonitoring.jetSelection_HT   = cms.string("pt > 30 && eta < 2.5")

#HLT_PFHT510
PFHT510_HTmonitoring = hltHTmonitoring.clone()
PFHT510_HTmonitoring.FolderName = cms.string('HLT/JME/HT/PFHT510/')
PFHT510_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT510_v*")
PFHT510_HTmonitoring.jetSelection_HT   = cms.string("pt > 30 && eta < 2.5")

#HLT_PFHT430
PFHT430_HTmonitoring = hltHTmonitoring.clone()
PFHT430_HTmonitoring.FolderName = cms.string('HLT/JME/HT/PFHT430/')
PFHT430_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT430_v*")
PFHT430_HTmonitoring.jetSelection_HT   = cms.string("pt > 30 && eta < 2.5")

#HLT_PFHT370
PFHT370_HTmonitoring = hltHTmonitoring.clone()
PFHT370_HTmonitoring.FolderName = cms.string('HLT/JME/HT/PFHT370/')
PFHT370_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT370_v*")
PFHT370_HTmonitoring.jetSelection_HT   = cms.string("pt > 30 && eta < 2.5")

#HLT_PFHT250
PFHT250_HTmonitoring = hltHTmonitoring.clone()
PFHT250_HTmonitoring.FolderName = cms.string('HLT/JME/HT/PFHT250/')
PFHT250_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT250_v*")
PFHT250_HTmonitoring.jetSelection_HT   = cms.string("pt > 30 && eta < 2.5")

#HLT_PFHT180
PFHT180_HTmonitoring = hltHTmonitoring.clone()
PFHT180_HTmonitoring.FolderName = cms.string('HLT/JME/HT/PFHT180/')
PFHT180_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT180_v*")
PFHT180_HTmonitoring.jetSelection_HT   = cms.string("pt > 30 && eta < 2.5")

# HLT_PFMETTypeOne110_PFMHT110_IDTight                                                                                                                              
PFMETTypeOne110_PFMHT110_HTmonitoring = hltHTmonitoring.clone()
PFMETTypeOne110_PFMHT110_HTmonitoring.FolderName = cms.string('HLT/EXO/MET/PFMETTypeOne110/')
PFMETTypeOne110_PFMHT110_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMETTypeOne110_PFMHT110_IDTight_v*")
PFMETTypeOne110_PFMHT110_HTmonitoring.jetSelection      = cms.string("pt > 100 && eta < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")
PFMETTypeOne110_PFMHT110_HTmonitoring.jetSelection_HT   = cms.string("pt > 30 && eta < 2.5")

# HLT_PFMETTypeOne120_PFMHT120_IDTight
PFMETTypeOne120_PFMHT120_HTmonitoring = hltHTmonitoring.clone()
PFMETTypeOne120_PFMHT120_HTmonitoring.FolderName = cms.string('HLT/EXO/MET/PFMETTypeOne120/')
PFMETTypeOne120_PFMHT120_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PPFMETTypeOne120_PFMHT120_IDTight_v*")
PFMETTypeOne120_PFMHT120_HTmonitoring.jetSelection      = cms.string("pt > 100 && eta < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")
PFMETTypeOne120_PFMHT120_HTmonitoring.jetSelection_HT   = cms.string("pt > 30 && eta < 2.5")

# HLT_PFMETTypeOne130_PFMHT130_IDTight
PFMETTypeOne130_PFMHT130_HTmonitoring = hltHTmonitoring.clone()
PFMETTypeOne130_PFMHT130_HTmonitoring.FolderName = cms.string('HLT/EXO/MET/PFMETTypeOne130/')
PFMETTypeOne130_PFMHT130_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMETTypeOne130_PFMHT130_IDTight_v*")
PFMETTypeOne130_PFMHT130_HTmonitoring.jetSelection      = cms.string("pt > 100 && eta < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")
PFMETTypeOne130_PFMHT130_HTmonitoring.jetSelection_HT   = cms.string("pt > 30 && eta < 2.5")

# HLT_PFMETTypeOne140_PFMHT140_IDTight
PFMETTypeOne140_PFMHT140_HTmonitoring = hltHTmonitoring.clone()
PFMETTypeOne140_PFMHT140_HTmonitoring.FolderName = cms.string('HLT/EXO/MET/PFMETTypeOne140/')
PFMETTypeOne140_PFMHT140_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMETTypeOne140_PFMHT140_IDTight_v*")
PFMETTypeOne140_PFMHT140_HTmonitoring.jetSelection      = cms.string("pt > 100 && eta < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")
PFMETTypeOne140_PFMHT140_HTmonitoring.jetSelection_HT   = cms.string("pt > 30 && eta < 2.5")

# HLT_PFMET120_PFMHT120_IDTight_PFHT60
PFMET120_PFMHT120_IDTight_PFHT60_HTmonitoring = hltHTmonitoring.clone()
PFMET120_PFMHT120_IDTight_PFHT60_HTmonitoring.FolderName = cms.string('HLT/EXO/MET/PFMET120_PFHT60/')
PFMET120_PFMHT120_IDTight_PFHT60_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_PFHT60_v*")
PFMET120_PFMHT120_IDTight_PFHT60_HTmonitoring.metSelection      = cms.string("pt > 220")
PFMET120_PFMHT120_IDTight_PFHT60_HTmonitoring.jetSelection      = cms.string("pt > 70 && abs(eta) < 2.4 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")

# HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60
PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_HTmonitoring = hltHTmonitoring.clone()
PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_HTmonitoring.FolderName = cms.string('HLT/EXO/MET/PFMETNoMu120_PFHT60/')
PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_v*")
PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_HTmonitoring.metSelection      = cms.string("pt > 220")
PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60_HTmonitoring.jetSelection      = cms.string("pt > 70 && abs(eta) < 2.4 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")

# HLT_PFMETTypeOne120_PFMHT120_IDTight_PFHT60
PFMETTypeOne120_PFMHT120_IDTight_PFHT60_HTmonitoring = hltHTmonitoring.clone()
PFMETTypeOne120_PFMHT120_IDTight_PFHT60_HTmonitoring.FolderName = cms.string('HLT/EXO/MET/PFMETTypeOne120_PFHT60/')
PFMETTypeOne120_PFMHT120_IDTight_PFHT60_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMETTypeOne120_PFMHT120_IDTight_PFHT60_v*")
PFMETTypeOne120_PFMHT120_IDTight_PFHT60_HTmonitoring.metSelection      = cms.string("pt > 220")
PFMETTypeOne120_PFMHT120_IDTight_PFHT60_HTmonitoring.jetSelection      = cms.string("pt > 70 && abs(eta) < 2.4 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")

exoHLTHTmonitoring = cms.Sequence(
    PFMETNoMu90_PFMHTNoMu90_HTmonitoring
    + PFMETNoMu120_PFMHTNoMu120_HTmonitoring
    + MET200_HTmonitoring
    + MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_HTmonitoring
    + MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_HTmonitoring
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

