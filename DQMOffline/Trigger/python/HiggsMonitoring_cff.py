import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.METMonitor_cfi import hltMETmonitoring
from DQMOffline.Trigger.TopMonitor_cfi import hltTOPmonitoring

# HLT_PFMET100_PFMHT100_IDTight_BTagCaloCSV_p05 MET monitoring
PFMET100_PFMHT100_IDTight_BTagCaloCSV_p05_METmonitoring = hltMETmonitoring.clone()
PFMET100_PFMHT100_IDTight_BTagCaloCSV_p05_METmonitoring.FolderName = cms.string('HLT/Higgs/PFMET100_BTag/')
PFMET100_PFMHT100_IDTight_BTagCaloCSV_p05_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET100_PFMHT100_IDTight_BTagCaloCSV_p05_v")
PFMET100_PFMHT100_IDTight_BTagCaloCSV_p05_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")

# HLT_PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05 MET monitoring
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_METmonitoring = hltMETmonitoring.clone()
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_METmonitoring.FolderName = cms.string('HLT/Higgs/PFMET110_BTag/')
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_v")
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")
# HLT_PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05 b-tag monitoring
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring = hltTOPmonitoring.clone()
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring.FolderName= cms.string('HLT/Higgs/PFMET110_BTag/')
# Selection
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring.leptJetDeltaRmin = cms.double(0.0)
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring.njets            = cms.uint32(1)
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring.jetSelection     = cms.string('pt>30 & abs(eta)<2.4')
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring.HTcut            = cms.double(0)
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring.nbjets           = cms.uint32(1)
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring.bjetSelection    = cms.string('pt>30 & abs(eta)<2.4')
PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring.workingpoint     = cms.double(0.8484) # Medium
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
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_METmonitoring.FolderName = cms.string('HLT/Higgs/PFMET120_BTag/')
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_v")
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")
# HLT_PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05 b-tag monitoring
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring = hltTOPmonitoring.clone()
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring.FolderName= cms.string('HLT/Higgs/PFMET120_BTag/')
# Selection
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring.leptJetDeltaRmin = cms.double(0.0)
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring.njets            = cms.uint32(1)
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring.jetSelection     = cms.string('pt>30 & abs(eta)<2.4')
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring.HTcut            = cms.double(0)
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring.nbjets           = cms.uint32(1)
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring.bjetSelection    = cms.string('pt>30 & abs(eta)<2.4')
PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring.workingpoint     = cms.double(0.8484) # Medium
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
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_METmonitoring.FolderName = cms.string('HLT/Higgs/PFMET130_BTag/')
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_v")
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")
# HLT_PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05 b-tag monitoring
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring = hltTOPmonitoring.clone()
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring.FolderName= cms.string('HLT/Higgs/PFMET130_BTag/')
# Selection
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring.leptJetDeltaRmin = cms.double(0.0)
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring.njets            = cms.uint32(1)
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring.jetSelection     = cms.string('pt>30 & abs(eta)<2.4')
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring.HTcut            = cms.double(0)
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring.nbjets           = cms.uint32(1)
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring.bjetSelection    = cms.string('pt>30 & abs(eta)<2.4')
PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring.workingpoint     = cms.double(0.8484) # Medium
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
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_METmonitoring.FolderName = cms.string('HLT/Higgs/PFMET140_BTag/')
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_v")
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_METmonitoring.jetSelection      = cms.string("pt > 100 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.8 && chargedHadronEnergyFraction > 0.1")
# HLT_PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05 b-tag monitoring
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring = hltTOPmonitoring.clone()
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.FolderName= cms.string('HLT/Higgs/PFMET140_BTag/')
# Selection
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.leptJetDeltaRmin = cms.double(0.0)
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.njets            = cms.uint32(1)
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.jetSelection     = cms.string('pt>30 & abs(eta)<2.4')
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.HTdefinition     = cms.string('pt>30 & abs(eta)<2.4')
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.HTcut            = cms.double(0)
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.nbjets           = cms.uint32(1)
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.bjetSelection    = cms.string('pt>30 & abs(eta)<2.4')
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.workingpoint     = cms.double(0.8484) # Medium
# Binning
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.histoPSet.htPSet = cms.PSet(nbins=cms.uint32(50), xmin=cms.double(0.0), xmax=cms.double(1000) )
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.histoPSet.jetPtBinning = cms.vdouble(0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,140,200,400)
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.histoPSet.HTBinning    = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700,900)
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.histoPSet.metBinning = cms.vdouble(0,20,40,60,80,100,125,150,175,200,300,400,500,700,900)
# Triggers
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_v')
PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFMET140_PFMHT140_IDTight_v')

higgsMonitorHLT = cms.Sequence(    
    PFMET100_PFMHT100_IDTight_BTagCaloCSV_p05_METmonitoring
    + PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_METmonitoring
    + PFMET110_PFMHT110_IDTight_BTagCaloCSV_p05_TOPmonitoring
    + PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_METmonitoring
    + PFMET120_PFMHT120_IDTight_BTagCaloCSV_p05_TOPmonitoring
    + PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_METmonitoring
    + PFMET130_PFMHT130_IDTight_BTagCaloCSV_p05_TOPmonitoring
    + PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_METmonitoring
    + PFMET140_PFMHT140_IDTight_BTagCaloCSV_p05_TOPmonitoring
)
