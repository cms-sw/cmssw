import FWCore.ParameterSet.Config as cms

process = cms.Process("EWKElectronSkim")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
    'rfio:/tmp/ikesisog/3E541A7B-EE86-DE11-BA46-001E682F273A.root'
    )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.GlobalTag.globaltag = cms.string('MC_31X_V3::All')


# HLT filter
import HLTrigger.HLTfilters.hltHighLevel_cfi
process.EWK_ElectronHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()

# Uncomment this to access 8E29 menu and filter on it
process.EWK_ElectronHLTFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT8E29")
process.EWK_ElectronHLTFilter.HLTPaths = ["HLT_Ele15_LW_L1R"]


# Electron filter
process.goodElectrons = cms.EDFilter("CandViewSelector",
  src = cms.InputTag("gsfElectrons"),
  cut = cms.string('pt > 30.0'),
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

EWK_ElectronSkimEventContent.outputCommands.extend(
      cms.untracked.vstring('drop *',
            "keep recoSuperClusters_*_*_*",
            "keep *_gsfElectrons_*_*",
            "keep recoGsfTracks_electronGsfTracks_*_*",
            "keep *_gsfElectronCores_*_*",
            "keep *_correctedHybridSuperClusters_*_*",
            "keep *_correctedMulti5x5SuperClustersWithPreshower_*_*",
            "keep edmTriggerResults_*_*_*",
            "keep recoCaloMETs_*_*_*",
            "keep recoMETs_*_*_*",
            "keep *_particleFlow_electrons_*",
            "keep *_pfMet_*_*",
            "keep *_multi5x5SuperClusterWithPreshower_*_*",
            "keep recoVertexs_*_*_*",
            "keep *_hltTriggerSummaryAOD_*_*",
            "keep floatedmValueMap_*_*_*",
            "keep recoBeamSpot_*_*_*" )
)

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
        filterName = cms.untracked.string('EWKSKIMESEL'),
        dataTier = cms.untracked.string('USER')
   ),
   fileName = cms.untracked.string('EWKElectronSkim.root')
)

process.outpath = cms.EndPath(process.EWK_ElectronSkimOutputModule)


