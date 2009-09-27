import FWCore.ParameterSet.Config as cms

process = cms.Process("EWKElectronSkim")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
    'file:/scratch1/cms/data/summer09/aodsim/zee/0022/E0869C04-088D-DE11-BFCA-001CC4A6CCE6.root'
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('MC_31X_V3::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

# HLT filter
import HLTrigger.HLTfilters.hltHighLevel_cfi
process.EWK_ElectronHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
# Uncomment this to access 8E29 menu and filter on it
#process.EWK_ElectronHLTFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT8E29")
#process.EWK_ElectronHLTFilter.HLTPaths = ["HLT_Ele10_LW_L1R", "HLT_DoubleEle5_SW_L1R"]
# Uncomment this to filter on 1E31 HLT menu
process.EWK_ElectronHLTFilter.HLTPaths = ["HLT_Ele20_SW_L1R", "HLT_DoubleEle10_SW_L1R"]

# Electron filter
process.goodElectrons = cms.EDFilter("CandViewSelector",
  src = cms.InputTag("gsfElectrons"),
  cut = cms.string('pt > 20'),
  filter = cms.bool(True)                                
)

# Skim path
process.EWK_ElectronSkimPath = cms.Path(
  process.EWK_ElectronHLTFilter +
  process.goodElectrons
)

# Output module configuration
from Configuration.EventContent.EventContent_cff import *
EWK_ElectronSkimEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
EWK_ElectronSkimEventContent.outputCommands.extend(AODEventContent.outputCommands)

EWK_ElectronSkimEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(
           'EWK_ElectronSkimPath')
    )
)

process.EWK_ElectronSkimOutputModule = cms.OutputModule("PoolOutputModule",
    EWK_ElectronSkimEventContent,
    EWK_ElectronSkimEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('EWKElectronSkim'),
        dataTier = cms.untracked.string('USER')
   ),
   fileName = cms.untracked.string('file:testEWKElectronSkim.root')
)

process.outpath = cms.EndPath(process.EWK_ElectronSkimOutputModule)


