import FWCore.ParameterSet.Config as cms

process = cms.Process("EWKHighPtMuSkim")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
    'file:/scratch2/users/fabozzi/spring10/zmm/38262142-DF46-DF11-8238-0030487C6A90.root'
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('GR_R_35X_V6::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

# Muon filter
process.goodMuons = cms.EDFilter("MuonSelector",
  src = cms.InputTag("muons"),
  cut = cms.string('pt > 15.0 && ( isGlobalMuon=1 || isTrackerMuon=1)'),
  filter = cms.bool(True)
)

# dxy filter on good muons
process.dxyFilteredMuons = cms.EDFilter("MuonSelector",
  src = cms.InputTag("goodMuons"),
  cut = cms.string('abs(innerTrack().dxy)<1.0'),
  filter = cms.bool(True)                                
)

# Skim path
process.EWK_HighPtMuSkimPath = cms.Path(
  process.goodMuons *
  process.dxyFilteredMuons
)

# Output module configuration
from Configuration.EventContent.EventContent_cff import *
EWK_MuSkimEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
EWK_MuSkimEventContent.outputCommands.extend(RECOEventContent.outputCommands)

EWK_MuSkimEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(
           'EWK_HighPtMuSkimPath')
    )
)

process.EWK_MuSkimOutputModule = cms.OutputModule("PoolOutputModule",
    EWK_MuSkimEventContent,
    EWK_MuSkimEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('EWKHighPtMuSkim'),
        dataTier = cms.untracked.string('USER')
   ),
   fileName = cms.untracked.string('EWK_HighPtMuSkim_SD_Mu.root')
)

process.outpath = cms.EndPath(process.EWK_MuSkimOutputModule)


