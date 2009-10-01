import FWCore.ParameterSet.Config as cms

process = cms.Process("EWKMuSkim")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
    'file:/scratch1/cms/data/summer09/aodsim/zmumu/0016/A68B6BD1-FF83-DE11-B579-001E68A99420.root'
#    'file:/scratch1/cms/data/summer09/aodsim/ppMuX/0010/9C519151-5883-DE11-8BC8-001AA0095119.root'
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('MC_31X_V3::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

# HLT filter
import HLTrigger.HLTfilters.hltHighLevel_cfi
process.EWK_MuHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
# Uncomment this to access 8E29 menu and filter on it
#process.EWK_MuHLTFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT8E29")
#process.EWK_MuHLTFilter.HLTPaths = ["HLT_Mu3", "HLT_DoubleMu3"]
# Uncomment this to filter on 1E31 HLT menu
process.EWK_MuHLTFilter.HLTPaths = ["HLT_Mu9", "HLT_DoubleMu3"]

# Muon filter
process.goodMuons = cms.EDFilter("CandViewSelector",
  src = cms.InputTag("muons"),
  cut = cms.string('isGlobalMuon = 1 & abs(eta) < 2.5 & pt > 0'),
  filter = cms.bool(True)                                
)

# Skim path
process.EWK_MuSkimPath = cms.Path(
  process.EWK_MuHLTFilter +
  process.goodMuons
)

# Output module configuration
from Configuration.EventContent.EventContent_cff import *
EWK_MuSkimEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
EWK_MuSkimEventContent.outputCommands.extend(AODEventContent.outputCommands)

EWK_MuSkimEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(
           'EWK_MuSkimPath')
    )
)

process.EWK_MuSkimOutputModule = cms.OutputModule("PoolOutputModule",
    EWK_MuSkimEventContent,
    EWK_MuSkimEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('EWKMuSkim'),
        dataTier = cms.untracked.string('USER')
   ),
   fileName = cms.untracked.string('file:testEWKMuSkim.root')
)

process.outpath = cms.EndPath(process.EWK_MuSkimOutputModule)


