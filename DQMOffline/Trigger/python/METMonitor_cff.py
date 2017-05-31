import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.METMonitor_cfi import hltMETmonitoring

# HLT_PFMET110_PFMHT110_IDTight
PFMET110_PFMHT110_IDTight_METmonitoring = hltMETmonitoring.clone()
PFMET110_PFMHT110_IDTight_METmonitoring.FolderName = cms.string('HLT/MET/PFMET110/')
PFMET110_PFMHT110_IDTight_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET110_PFMHT110_IDTight_v*")
PFMET110_PFMHT110_IDTight_METmonitoring.jetSelection      = cms.string("pt[0] > 100 && eta[0] < 2.5 && neutralHadronEnergyFraction[0] < 0.8 && chargedHadronEnergyFraction[0] > 0.1")
PFMET110_PFMHT110_IDTight_METmonitoring.muoSelection      = cms.string("isTightMuon()")

#HLT_PFMET120_PFMHT120_IDTight
PFMET120_PFMHT120_IDTight_METmonitoring = hltMETmonitoring.clone()
PFMET120_PFMHT120_IDTight_METmonitoring.FolderName = cms.string('HLT/MET/PFMET120')
PFMET120_PFMHT120_IDTight_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v*")
PFMET120_PFMHT120_IDTight_METmonitoring.jetSelection      = cms.string("pt[0] > 100 && eta[0] < 2.5 && neutralHadronEnergyFraction[0] < 0.8 && chargedHadronEnergyFraction[0] > 0.1")
PFMET120_PFMHT120_IDTight_METmonitoring.muoSelection      = cms.string("isTightMuon()")

# HLT_PFMET130_PFMHT130_IDTight
PFMET130_PFMHT130_IDTight_METmonitoring = hltMETmonitoring.clone()
PFMET130_PFMHT130_IDTight_METmonitoring.FolderName = cms.string('HLT/MET/PFMET130/')
PFMET130_PFMHT130_IDTight_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET130_PFMHT130_IDTight_v*")
PFMET130_PFMHT130_IDTight_METmonitoring.jetSelection      = cms.string("pt[0] > 100 && eta[0] < 2.5 && neutralHadronEnergyFraction[0] < 0.8 && chargedHadronEnergyFraction[0] > 0.1")
PFMET130_PFMHT130_IDTight_METmonitoring.muoSelection      = cms.string("isTightMuon()")

# HLT_PFMET140_PFMHT140_IDTight
PFMET140_PFMHT140_IDTight_METmonitoring = hltMETmonitoring.clone()
PFMET140_PFMHT140_IDTight_METmonitoring.FolderName = cms.string('HLT/MET/PFMET140/')
PFMET140_PFMHT140_IDTight_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET140_PFMHT140_IDTight_v*")
PFMET140_PFMHT140_IDTight_METmonitoring.jetSelection      = cms.string("pt[0] > 100 && eta[0] < 2.5 && neutralHadronEnergyFraction[0] < 0.8 && chargedHadronEnergyFraction[0] > 0.1")
PFMET140_PFMHT140_IDTight_METmonitoring.muoSelection      = cms.string("isTightMuon()")

# HLT_PFMETTypeOne110_PFMHT110_IDTight                                                                                                                              
PFMETTypeOne110_PFMHT110_METmonitoring = hltMETmonitoring.clone()
PFMETTypeOne110_PFMHT110_METmonitoring.FolderName = cms.string('HLT/MET/PFMETTypeOne110/')
PFMETTypeOne110_PFMHT110_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMETTypeOne110_PFMHT110_IDTight_v*")
PFMETTypeOne110_PFMHT110_METmonitoring.jetSelection      = cms.string("pt[0] > 100 && eta[0] < 2.5 && neutralHadronEnergyFraction[0] < 0.8 && chargedHadronEnergyFraction[0] > 0.1")
PFMETTypeOne110_PFMHT110_METmonitoring.muoSelection      = cms.string("isTightMuon()")

# HLT_PFMETTypeOne120_PFMHT120_IDTight
PFMETTypeOne120_PFMHT120_METmonitoring = hltMETmonitoring.clone()
PFMETTypeOne120_PFMHT120_METmonitoring.FolderName = cms.string('HLT/MET/PFMETTypeOne120/')
PFMETTypeOne120_PFMHT120_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PPFMETTypeOne120_PFMHT120_IDTight_v*")
PFMETTypeOne120_PFMHT120_METmonitoring.jetSelection      = cms.string("pt[0] > 100 && eta[0] < 2.5 && neutralHadronEnergyFraction[0] < 0.8 && chargedHadronEnergyFraction[0] > 0.1")
PFMETTypeOne120_PFMHT120_METmonitoring.muoSelection      = cms.string("isTightMuon()")

# HLT_PFMETTypeOne130_PFMHT130_IDTight
PFMETTypeOne130_PFMHT130_METmonitoring = hltMETmonitoring.clone()
PFMETTypeOne130_PFMHT130_METmonitoring.FolderName = cms.string('HLT/MET/PFMETTypeOne130/')
PFMETTypeOne130_PFMHT130_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMETTypeOne130_PFMHT130_IDTight_v*")
PFMETTypeOne130_PFMHT130_METmonitoring.jetSelection      = cms.string("pt[0] > 100 && eta[0] < 2.5 && neutralHadronEnergyFraction[0] < 0.8 && chargedHadronEnergyFraction[0] > 0.1")
PFMETTypeOne130_PFMHT130_METmonitoring.muoSelection      = cms.string("isTightMuon()")

# HLT_PFMETTypeOne140_PFMHT140_IDTight
PFMETTypeOne140_PFMHT140_METmonitoring = hltMETmonitoring.clone()
PFMETTypeOne140_PFMHT140_METmonitoring.FolderName = cms.string('HLT/MET/PFMETTypeOne140/')
PFMETTypeOne140_PFMHT140_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMETTypeOne140_PFMHT140_IDTight_v*")
PFMETTypeOne140_PFMHT140_METmonitoring.jetSelection      = cms.string("pt[0] > 100 && eta[0] < 2.5 && neutralHadronEnergyFraction[0] < 0.8 && chargedHadronEnergyFraction[0] > 0.1")
PFMETTypeOne140_PFMHT140_METmonitoring.muoSelection      = cms.string("isTightMuon()")

# HLT_PFMETNoMu90_PFMHTNoMu90_IDTight
PFMETNoMu90_PFMHTNoMu90_METmonitoring = hltMETmonitoring.clone()
PFMETNoMu90_PFMHTNoMu90_METmonitoring.FolderName = cms.string('HLT/MET/PFMETNoMu90/')
PFMETNoMu90_PFMHTNoMu90_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMETNoMu90_PFMHTNoMu90_IDTight_v*")

# HLT_PFMETNoMu110_PFMHTNoMu110_IDTight
PFMETNoMu110_PFMHTNoMu110_METmonitoring = hltMETmonitoring.clone()
PFMETNoMu110_PFMHTNoMu110_METmonitoring.FolderName = cms.string('HLT/MET/PFMETNoMu110/')
PFMETNoMu110_PFMHTNoMu110_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_v*")
PFMETNoMu110_PFMHTNoMu110_METmonitoring.jetSelection      = cms.string("pt[0] > 100 && eta[0] < 2.5 && neutralHadronEnergyFraction[0] < 0.8 && chargedHadronEnergyFraction[0] > 0.1")
PFMETNoMu110_PFMHTNoMu110_METmonitoring.muoSelection      = cms.string("isTightMuon()")

# HLT_PFMETNoMu120_PFMHTNoMu120_IDTight
PFMETNoMu120_PFMHTNoMu120_METmonitoring = hltMETmonitoring.clone()
PFMETNoMu120_PFMHTNoMu120_METmonitoring.FolderName = cms.string('HLT/MET/PFMETNoMu120/')
PFMETNoMu120_PFMHTNoMu120_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")
PFMETNoMu120_PFMHTNoMu120_METmonitoring.jetSelection      = cms.string("pt[0] > 100 && eta[0] < 2.5 && neutralHadronEnergyFraction[0] < 0.8 && chargedHadronEnergyFraction[0] > 0.1")
PFMETNoMu120_PFMHTNoMu120_METmonitoring.muoSelection      = cms.string("isTightMuon()")

# HLT_PFMETNoMu130_PFMHTNoMu130_IDTight
PFMETNoMu130_PFMHTNoMu130_METmonitoring = hltMETmonitoring.clone()
PFMETNoMu130_PFMHTNoMu130_METmonitoring.FolderName = cms.string('HLT/MET/PFMETNoMu130/')
PFMETNoMu130_PFMHTNoMu130_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("_v*")
PFMETNoMu130_PFMHTNoMu130_METmonitoring.jetSelection      = cms.string("pt[0] > 100 && eta[0] < 2.5 && neutralHadronEnergyFraction[0] < 0.8 && chargedHadronEnergyFraction[0] > 0.1")
PFMETNoMu130_PFMHTNoMu130_METmonitoring.muoSelection      = cms.string("isTightMuon()")

# HLT_PFMETNoMu140_PFMHTNoMu140_IDTight
PFMETNoMu140_PFMHTNoMu140_METmonitoring = hltMETmonitoring.clone()
PFMETNoMu140_PFMHTNoMu140_METmonitoring.FolderName = cms.string('HLT/MET/PFMETNoMu140/')
PFMETNoMu140_PFMHTNoMu140_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("_v*")
PFMETNoMu140_PFMHTNoMu140_METmonitoring.jetSelection      = cms.string("pt[0] > 100 && eta[0] < 2.5 && neutralHadronEnergyFraction[0] < 0.8 && chargedHadronEnergyFraction[0] > 0.1")
PFMETNoMu140_PFMHTNoMu140_METmonitoring.muoSelection      = cms.string("isTightMuon()")

# HLT_MET200
MET200_METmonitoring = hltMETmonitoring.clone()
MET200_METmonitoring.FolderName = cms.string('HLT/MET/MET200/')
MET200_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_MET200_v*")

# HLT_MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_IDTight
MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_METmonitoring = hltMETmonitoring.clone()
MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_METmonitoring.FolderName = cms.string('HLT/MET/MonoCentralPFJet80_PFMETNoMu110/')
MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_IDTight_v*")
MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_METmonitoring.jetSelection      = cms.string("pt[0] > 100 && eta[0] < 2.5 && neutralHadronEnergyFraction[0] < 0.8 && chargedHadronEnergyFraction[0] > 0.1")
MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_METmonitoring.muoSelection      = cms.string("isTightMuon()")

# HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight
MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_METmonitoring = hltMETmonitoring.clone()
MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_METmonitoring.FolderName = cms.string('HLT/MET/MonoCentralPFJet80_PFMETNoMu120/')
MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")
MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_METmonitoring.jetSelection      = cms.string("pt[0] > 100 && eta[0] < 2.5 && neutralHadronEnergyFraction[0] < 0.8 && chargedHadronEnergyFraction[0] > 0.1")
MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_METmonitoring.muoSelection      = cms.string("isTightMuon()")

# HLT_MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_IDTight
MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_METmonitoring = hltMETmonitoring.clone()
MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_METmonitoring.FolderName = cms.string('HLT/MET/MonoCentralPFJet80_PFMETNoMu130/')
MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_IDTight_v*")
MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_METmonitoring.jetSelection      = cms.string("pt[0] > 100 && eta[0] < 2.5 && neutralHadronEnergyFraction[0] < 0.8 && chargedHadronEnergyFraction[0] > 0.1")
MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_METmonitoring.muoSelection      = cms.string("isTightMuon()")

# HLT_MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_IDTight
MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_METmonitoring = hltMETmonitoring.clone()
MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_METmonitoring.FolderName = cms.string('HLT/MET/MonoCentralPFJet80_PFMETNoMu140/')
MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("_v*")
MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_METmonitoring.jetSelection      = cms.string("pt[0] > 100 && eta[0] < 2.5 && neutralHadronEnergyFraction[0] < 0.8 && chargedHadronEnergyFraction[0] > 0.1")
MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_METmonitoring.muoSelection      = cms.string("isTightMuon()")

# HLT_MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_IDTight
MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_METmonitoring = hltMETmonitoring.clone()
MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_METmonitoring.FolderName = cms.string('HLT/MET/MonoCentralPFJet80_PFMETNoMu90/')
MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_IDTight_v*")
MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_METmonitoring.jetSelection      = cms.string("pt[0] > 100 && eta[0] < 2.5 && neutralHadronEnergyFraction[0] < 0.8 && chargedHadronEnergyFraction[0] > 0.1")
MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_METmonitoring.muoSelection      = cms.string("isTightMuon()")

# HLT_PFHT500_PFMET100_PFMHT100_IDTight
PFHT500_PFMET100_PFMHT100_METmonitoring = hltMETmonitoring.clone()
PFHT500_PFMET100_PFMHT100_METmonitoring.FolderName = cms.string('HLT/MET/PFHT500_PFMET100_PFMHT100/')
PFHT500_PFMET100_PFMHT100_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT500_PFMET100_PFMHT100_IDTight_v*")

# HLT_PFHT500_PFMET110_PFMHT110_IDTight
PFHT500_PFMET110_PFMHT110_METmonitoring = hltMETmonitoring.clone()
PFHT500_PFMET110_PFMHT110_METmonitoring.FolderName = cms.string('HLT/MET/PFHT500_PFMET110_PFMHT110/')
PFHT500_PFMET110_PFMHT110_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT500_PFMET110_PFMHT110_IDTight_v*")

# HLT_PFHT700_PFMET85_PFMHT85_IDTight
PFHT700_PFMET85_PFMHT85_METmonitoring = hltMETmonitoring.clone()
PFHT700_PFMET85_PFMHT85_METmonitoring.FolderName = cms.string('HLT/MET/PFHT700_PFMET85_PFMHT85/')
PFHT700_PFMET85_PFMHT85_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT700_PFMET85_PFMHT85_IDTight_v*")

# HLT_PFHT700_PFMET95_PFMHT95_IDTight
PFHT700_PFMET95_PFMHT95_METmonitoring = hltMETmonitoring.clone()
PFHT700_PFMET95_PFMHT95_METmonitoring.FolderName = cms.string('HLT/MET/PFHT700_PFMET95_PFMHT95/')
PFHT700_PFMET95_PFMHT95_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT700_PFMET95_PFMHT95_IDTight_v*")

# HLT_PFHT800_PFMET75_PFMHT75_IDTight
PFHT800_PFMET75_PFMHT75_METmonitoring = hltMETmonitoring.clone()
PFHT800_PFMET75_PFMHT75_METmonitoring.FolderName = cms.string('HLT/MET/PFHT800_PFMET75_PFMHT75/')
PFHT800_PFMET75_PFMHT75_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT800_PFMET75_PFMHT75_IDTight_v*")

# HLT_PFHT800_PFMET85_PFMHT85_IDTight
PFHT800_PFMET85_PFMHT85_METmonitoring = hltMETmonitoring.clone()
PFHT800_PFMET85_PFMHT85_METmonitoring.FolderName = cms.string('HLT/MET/PFHT800_PFMET85_PFMHT85/')
PFHT800_PFMET85_PFMHT85_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT800_PFMET85_PFMHT85_IDTight_v*")

#HLT_PFHT890
HLT_PFHT890_METmonitoring = hltMETmonitoring.clone()
HLT_PFHT890_METmonitoring.FolderName = cms.string('HLT/MET/HLT_PFHT890/')
HLT_PFHT890_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT890_v*")

#HLT_PFHT780
HLT_PFHT780_METmonitoring = hltMETmonitoring.clone()
HLT_PFHT780_METmonitoring.FolderName = cms.string('HLT/MET/HLT_PFHT780/')
HLT_PFHT780_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT780_v*")

#HLT_PFHT680
HLT_PFHT680_METmonitoring = hltMETmonitoring.clone()
HLT_PFHT680_METmonitoring.FolderName = cms.string('HLT/MET/HLT_PFHT680/')
HLT_PFHT680_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT680_v*")

#HLT_PFHT590
HLT_PFHT590_METmonitoring = hltMETmonitoring.clone()
HLT_PFHT590_METmonitoring.FolderName = cms.string('HLT/MET/HLT_PFHT590/')
HLT_PFHT590_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT590_v*")

#HLT_PFHT510
HLT_PFHT510_METmonitoring = hltMETmonitoring.clone()
HLT_PFHT510_METmonitoring.FolderName = cms.string('HLT/MET/HLT_PFHT510/')
HLT_PFHT510_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT510_v*")

#HLT_PFHT430
HLT_PFHT430_METmonitoring = hltMETmonitoring.clone()
HLT_PFHT430_METmonitoring.FolderName = cms.string('HLT/MET/HLT_PFHT430/')
HLT_PFHT430_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT430_v*")

#HLT_PFHT370
HLT_PFHT370_METmonitoring = hltMETmonitoring.clone()
HLT_PFHT370_METmonitoring.FolderName = cms.string('HLT/MET/HLT_PFHT370/')
HLT_PFHT370_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT370_v*")

#HLT_PFHT250
HLT_PFHT250_METmonitoring = hltMETmonitoring.clone()
HLT_PFHT250_METmonitoring.FolderName = cms.string('HLT/MET/HLT_PFHT250/')
HLT_PFHT250_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT250_v*")

#HLT_PFHT180
HLT_PFHT180_METmonitoring = hltMETmonitoring.clone()
HLT_PFHT180_METmonitoring.FolderName = cms.string('HLT/MET/HLT_PFHT180/')
HLT_PFHT180_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT180_v*")

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
    + PFHT890_METmonitoring
    + PFHT780_METmonitoring
    + PFHT680_METmonitoring
    + PFHT590_METmonitoring
    + PFHT510_METmonitoring
    + PFHT430_METmonitoring
    + PFHT370_METmonitoring
    + PFHT250_METmonitoring
    + PFHT180_METmonitoring
)

