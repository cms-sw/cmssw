import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.METMonitor_cfi import hltMETmonitoring

# HLT_PFMET110_PFMHT110_IDTight
PFMET110_PFMHT110_IDTight_METmonitoring = hltMETmonitoring.clone()
PFMET110_PFMHT110_IDTight_METmonitoring.FolderName = cms.string('HLT/MET/PFMET110/')
PFMET110_PFMHT110_IDTight_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET110_PFMHT110_IDTight_v")
PFMET110_PFMHT110_IDTight_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")

#HLT_PFMET120_PFMHT120_IDTight
PFMET120_PFMHT120_IDTight_METmonitoring = hltMETmonitoring.clone()
PFMET120_PFMHT120_IDTight_METmonitoring.FolderName = cms.string('HLT/MET/PFMET120')
PFMET120_PFMHT120_IDTight_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v")
PFMET120_PFMHT120_IDTight_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")

# HLT_PFMET130_PFMHT130_IDTight
PFMET130_PFMHT130_IDTight_METmonitoring = hltMETmonitoring.clone()
PFMET130_PFMHT130_IDTight_METmonitoring.FolderName = cms.string('HLT/MET/PFMET130/')
PFMET130_PFMHT130_IDTight_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET130_PFMHT130_IDTight_v")
PFMET130_PFMHT130_IDTight_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")

# HLT_PFMET140_PFMHT140_IDTight
PFMET140_PFMHT140_IDTight_METmonitoring = hltMETmonitoring.clone()
PFMET140_PFMHT140_IDTight_METmonitoring.FolderName = cms.string('HLT/MET/PFMET140/')
PFMET140_PFMHT140_IDTight_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET140_PFMHT140_IDTight_v")
PFMET140_PFMHT140_IDTight_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")

# HLT_PFMETTypeOne110_PFMHT110_IDTight                                                                                                                              
PFMETTypeOne110_PFMHT110_METmonitoring = hltMETmonitoring.clone()
PFMETTypeOne110_PFMHT110_METmonitoring.FolderName = cms.string('HLT/MET/PFMETTypeOne110/')
PFMETTypeOne110_PFMHT110_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMETTypeOne110_PFMHT110_IDTight_v")
PFMETTypeOne110_PFMHT110_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")

# HLT_PFMETTypeOne120_PFMHT120_IDTight
PFMETTypeOne120_PFMHT120_METmonitoring = hltMETmonitoring.clone()
PFMETTypeOne120_PFMHT120_METmonitoring.FolderName = cms.string('HLT/MET/PFMETTypeOne120/')
PFMETTypeOne120_PFMHT120_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PPFMETTypeOne120_PFMHT120_IDTight_v")
PFMETTypeOne120_PFMHT120_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")

# HLT_PFMETTypeOne130_PFMHT130_IDTight
PFMETTypeOne130_PFMHT130_METmonitoring = hltMETmonitoring.clone()
PFMETTypeOne130_PFMHT130_METmonitoring.FolderName = cms.string('HLT/MET/PFMETTypeOne130/')
PFMETTypeOne130_PFMHT130_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMETTypeOne130_PFMHT130_IDTight_v")
PFMETTypeOne130_PFMHT130_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")

# HLT_PFMETTypeOne140_PFMHT140_IDTight
PFMETTypeOne140_PFMHT140_METmonitoring = hltMETmonitoring.clone()
PFMETTypeOne140_PFMHT140_METmonitoring.FolderName = cms.string('HLT/MET/PFMETTypeOne140/')
PFMETTypeOne140_PFMHT140_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMETTypeOne140_PFMHT140_IDTight_v")
PFMETTypeOne140_PFMHT140_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")

# HLT_PFMETNoMu90_PFMHTNoMu90_IDTight
PFMETNoMu90_PFMHTNoMu90_METmonitoring = hltMETmonitoring.clone()
PFMETNoMu90_PFMHTNoMu90_METmonitoring.FolderName = cms.string('HLT/MET/PFMETNoMu90/')
PFMETNoMu90_PFMHTNoMu90_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMETNoMu90_PFMHTNoMu90_IDTight_v")
PFMETNoMu90_PFMHTNoMu90_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")

# HLT_PFMETNoMu110_PFMHTNoMu110_IDTight
PFMETNoMu110_PFMHTNoMu110_METmonitoring = hltMETmonitoring.clone()
PFMETNoMu110_PFMHTNoMu110_METmonitoring.FolderName = cms.string('HLT/MET/PFMETNoMu110/')
PFMETNoMu110_PFMHTNoMu110_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMETNoMu110_PFMHTNoMu110_IDTight_v")
PFMETNoMu110_PFMHTNoMu110_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")

# HLT_PFMETNoMu120_PFMHTNoMu120_IDTight
PFMETNoMu120_PFMHTNoMu120_METmonitoring = hltMETmonitoring.clone()
PFMETNoMu120_PFMHTNoMu120_METmonitoring.FolderName = cms.string('HLT/MET/PFMETNoMu120/')
PFMETNoMu120_PFMHTNoMu120_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v")
PFMETNoMu120_PFMHTNoMu120_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")

# HLT_PFMETNoMu130_PFMHTNoMu130_IDTight
PFMETNoMu130_PFMHTNoMu130_METmonitoring = hltMETmonitoring.clone()
PFMETNoMu130_PFMHTNoMu130_METmonitoring.FolderName = cms.string('HLT/MET/PFMETNoMu130/')
PFMETNoMu130_PFMHTNoMu130_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v")
PFMETNoMu130_PFMHTNoMu130_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")

# HLT_PFMETNoMu140_PFMHTNoMu140_IDTight
PFMETNoMu140_PFMHTNoMu140_METmonitoring = hltMETmonitoring.clone()
PFMETNoMu140_PFMHTNoMu140_METmonitoring.FolderName = cms.string('HLT/MET/PFMETNoMu140/')
PFMETNoMu140_PFMHTNoMu140_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v")
PFMETNoMu140_PFMHTNoMu140_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")

# HLT_MET200
MET200_METmonitoring = hltMETmonitoring.clone()
MET200_METmonitoring.FolderName = cms.string('HLT/MET/MET200/')
MET200_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_MET200_v")

# HLT_MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_IDTight
MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_METmonitoring = hltMETmonitoring.clone()
MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_METmonitoring.FolderName = cms.string('HLT/MET/MonoCentralPFJet80_PFMETNoMu110/')
MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_IDTight")
MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")

# HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight
MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_METmonitoring = hltMETmonitoring.clone()
MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_METmonitoring.FolderName = cms.string('HLT/MET/MonoCentralPFJet80_PFMETNoMu120/')
MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v")
MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")

# HLT_MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_IDTight
MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_METmonitoring = hltMETmonitoring.clone()
MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_METmonitoring.FolderName = cms.string('HLT/MET/MonoCentralPFJet80_PFMETNoMu130/')
MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_IDTight_v")
MonoCentralPFJet80_PFMETNoMu130_PFMHTNoMu130_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")

# HLT_MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_IDTight
MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_METmonitoring = hltMETmonitoring.clone()
MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_METmonitoring.FolderName = cms.string('HLT/MET/MonoCentralPFJet80_PFMETNoMu140/')
MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_IDTight_v")
MonoCentralPFJet80_PFMETNoMu140_PFMHTNoMu140_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")

# HLT_MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_IDTight
MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_METmonitoring = hltMETmonitoring.clone()
MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_METmonitoring.FolderName = cms.string('HLT/MET/MonoCentralPFJet80_PFMETNoMu90/')
MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_IDTight_v")
MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")

# HLT_PFHT500_PFMET100_PFMHT100_IDTight
PFHT500_PFMET100_PFMHT100_METmonitoring = hltMETmonitoring.clone()
PFHT500_PFMET100_PFMHT100_METmonitoring.FolderName = cms.string('HLT/MET/PFHT500_PFMET100_PFMHT100/')
PFHT500_PFMET100_PFMHT100_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT500_PFMET100_PFMHT100_IDTight_v")

# HLT_PFHT500_PFMET110_PFMHT110_IDTight
PFHT500_PFMET110_PFMHT110_METmonitoring = hltMETmonitoring.clone()
PFHT500_PFMET110_PFMHT110_METmonitoring.FolderName = cms.string('HLT/MET/PFHT500_PFMET110_PFMHT110/')
PFHT500_PFMET110_PFMHT110_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT500_PFMET110_PFMHT110_IDTight_v")

# HLT_PFHT700_PFMET85_PFMHT85_IDTight
PFHT700_PFMET85_PFMHT85_METmonitoring = hltMETmonitoring.clone()
PFHT700_PFMET85_PFMHT85_METmonitoring.FolderName = cms.string('HLT/MET/PFHT700_PFMET85_PFMHT85/')
PFHT700_PFMET85_PFMHT85_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT700_PFMET85_PFMHT85_IDTight_v")

# HLT_PFHT700_PFMET95_PFMHT95_IDTight
PFHT700_PFMET95_PFMHT95_METmonitoring = hltMETmonitoring.clone()
PFHT700_PFMET95_PFMHT95_METmonitoring.FolderName = cms.string('HLT/MET/PFHT700_PFMET95_PFMHT95/')
PFHT700_PFMET95_PFMHT95_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT700_PFMET95_PFMHT95_IDTight_v")

# HLT_PFHT800_PFMET75_PFMHT75_IDTight
PFHT800_PFMET75_PFMHT75_METmonitoring = hltMETmonitoring.clone()
PFHT800_PFMET75_PFMHT75_METmonitoring.FolderName = cms.string('HLT/MET/PFHT800_PFMET75_PFMHT75/')
PFHT800_PFMET75_PFMHT75_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT800_PFMET75_PFMHT75_IDTight_v")

# HLT_PFHT800_PFMET85_PFMHT85_IDTight
PFHT800_PFMET85_PFMHT85_METmonitoring = hltMETmonitoring.clone()
PFHT800_PFMET85_PFMHT85_METmonitoring.FolderName = cms.string('HLT/MET/PFHT800_PFMET85_PFMHT85/')
PFHT800_PFMET85_PFMHT85_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT800_PFMET85_PFMHT85_IDTight_v")


from DQMOffline.Trigger.TopMonitor_cfi import hltTOPmonitoring

# HLT_PFMET100_PFMHT100_IDTight_BTagCaloCSV_p05 MET monitoring
PFMET100_PFMHT100_IDTight_BTagCaloCSV_p05_METmonitoring = hltMETmonitoring.clone()
PFMET100_PFMHT100_IDTight_BTagCaloCSV_p05_METmonitoring.FolderName = cms.string('HLT/MET/PFMET100_BTag/')
PFMET100_PFMHT100_IDTight_BTagCaloCSV_p05_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET100_PFMHT100_IDTight_BTagCaloCSV_p05_v")
PFMET100_PFMHT100_IDTight_BTagCaloCSV_p05_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")

# HLT_PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05 MET monitoring
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_METmonitoring = hltMETmonitoring.clone()
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_METmonitoring.FolderName = cms.string('HLT/MET/PFMET110_BTag/')
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_v")
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")
# HLT_PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05 b-tag monitoring
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring = hltTOPmonitoring.clone()
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring.FolderName= cms.string('HLT/MET/PFMET110_BTag/')
# Selection
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring.leptJetDeltaRmin = cms.double(0.0)
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring.njets            = cms.uint32(1)
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring.jetSelection     = cms.string('pt>30 & abs(eta)<2.4')
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring.HTcut            = cms.double(0)
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring.nbjets           = cms.uint32(1)
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring.bjetSelection    = cms.string('pt>30 & abs(eta)<2.4')
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring.workingpoint     = cms.double(0.8484) # medium
# Binning
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring.histoPSet.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400)
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring.histoPSet.HTBinning    = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700,900)
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring.histoPSet.metBinning = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700,900)
# Triggers
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_v')
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFMET110_PFMHT110_IDTight_v')


# HLT_PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05 MET monitoring
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_METmonitoring = hltMETmonitoring.clone()
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_METmonitoring.FolderName = cms.string('HLT/MET/PFMET120_BTag/')
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_v")
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")
# HLT_PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05 b-tag monitoring
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring = hltTOPmonitoring.clone()
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring.FolderName= cms.string('HLT/MET/PFMET120_BTag/')
# Selection
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring.leptJetDeltaRmin = cms.double(0.0)
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring.njets            = cms.uint32(1)
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring.jetSelection     = cms.string('pt>30 & abs(eta)<2.4')
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring.HTcut            = cms.double(0)
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring.nbjets           = cms.uint32(1)
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring.bjetSelection    = cms.string('pt>30 & abs(eta)<2.4')
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring.workingpoint     = cms.double(0.8484) # medium
# Binning
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring.histoPSet.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,200,400)
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring.histoPSet.HTBinning    = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700,900)
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring.histoPSet.metBinning = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700,900)
# Triggers
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_v')
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFMET120_PFMHT120_IDTight_v')


# HLT_PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05 MET monitoring
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_METmonitoring = hltMETmonitoring.clone()
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_METmonitoring.FolderName = cms.string('HLT/MET/PFMET130_BTag/')
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_v")
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")
# HLT_PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05 b-tag monitoring
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring = hltTOPmonitoring.clone()
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring.FolderName= cms.string('HLT/MET/PFMET130_BTag/')
# Selection
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring.leptJetDeltaRmin = cms.double(0.0)
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring.njets            = cms.uint32(1)
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring.jetSelection     = cms.string('pt>30 & abs(eta)<2.4')
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring.HTcut            = cms.double(0)
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring.nbjets           = cms.uint32(1)
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring.bjetSelection    = cms.string('pt>30 & abs(eta)<2.4')
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring.workingpoint     = cms.double(0.8484) # medium
# Binning
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring.histoPSet.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,130,200,400)
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring.histoPSet.HTBinning    = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700,900)
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring.histoPSet.metBinning = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700,900)
# Triggers
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_v')
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFMET130_PFMHT130_IDTight_v')


# HLT_PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05 MET monitoring
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_METmonitoring = hltMETmonitoring.clone()
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_METmonitoring.FolderName = cms.string('HLT/MET/PFMET140_BTag/')
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_v")
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")
# HLT_PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05 b-tag monitoring
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring = hltTOPmonitoring.clone()
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.FolderName= cms.string('HLT/MET/PFMET140_BTag/')
# Selection
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.leptJetDeltaRmin = cms.double(0.0)
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.njets            = cms.uint32(1)
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.jetSelection     = cms.string('pt>30 & abs(eta)<2.4')
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.HTcut            = cms.double(0)
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.nbjets           = cms.uint32(1)
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.bjetSelection    = cms.string('pt>30 & abs(eta)<2.4')
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.workingpoint     = cms.double(0.8484) # medium
# Binning
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.histoPSet.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,140,200,400)
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.histoPSet.HTBinning    = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700,900)
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.histoPSet.metBinning = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700,900)
# Triggers
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_v')
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFMET140_PFMHT140_IDTight_v')


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
    + PFMET100_PFMHT100_IDTight_BTagCaloCSV_p05_METmonitoring
    + PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_METmonitoring
    + PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring
    + PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_METmonitoring
    + PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring
    + PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_METmonitoring
    + PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring
    + PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_METmonitoring
    + PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring
)

