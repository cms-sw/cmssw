import FWCore.ParameterSet.Config as cms

process = cms.Process("EWKDimuonSkim")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
    'file:/scratch1/cms/data/summer09/aodsim/zmumu/0016/A68B6BD1-FF83-DE11-B579-001E68A99420.root'
#    'file:/scratch1/cms/data/summer09/aodsim/ppMuX/0010/9C519151-5883-DE11-8BC8-001AA0095119.root'
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(500) )

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
process.EWK_MuHLTFilter.HLTPaths = ["HLT_Mu9"]

# Muon candidates filters 
process.goodAODMuons = cms.EDFilter("CandViewSelector",
  src = cms.InputTag("muons"),
  cut = cms.string('pt > 0'),
  filter = cms.bool(True)                                
)

process.goodAODGlobalMuons = cms.EDFilter("CandViewSelector",
  src = cms.InputTag("goodAODMuons"),
  cut = cms.string('isGlobalMuon=1'),
  filter = cms.bool(True)                                
)

# Track candidates
process.trackCandsUnfiltered = cms.EDProducer("ConcreteChargedCandidateProducer",
    src          = cms.InputTag("generalTracks"),
    particleType = cms.string('mu+')   # to fix mass hypothesis
)

# Track candidates filter
process.goodAODTrackCands = cms.EDFilter("CandViewSelector",
    src = cms.InputTag("trackCandsUnfiltered"),
    cut = cms.string('pt > 10')
)

# Dimuon candidates

process.dimuonsAOD = cms.EDFilter("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(False),
    cut = cms.string('mass > 0'),
    decay = cms.string("goodAODMuons@+ goodAODMuons@-")
)

process.dimuonsGlobalAOD = cms.EDFilter("CandViewRefSelector",
    src = cms.InputTag("dimuonsAOD"),
    cut = cms.string('charge = 0 & daughter(0).isGlobalMuon = 1 & daughter(1).isGlobalMuon = 1')
)

process.dimuonsOneStandAloneMuonAOD = cms.EDFilter("CandViewRefSelector",
    src = cms.InputTag("dimuonsAOD"),
    cut = cms.string('charge = 0 & mass > 20 & ( (daughter(0).isStandAloneMuon = 1 & daughter(0).isGlobalMuon = 0 & daughter(1).isGlobalMuon = 1) | (daughter(1).isStandAloneMuon = 1 & daughter(1).isGlobalMuon = 0 & daughter(0).isGlobalMuon = 1) )')
)

process.dimuonsOneTrackAOD = cms.EDFilter("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(False),
    cut = cms.string('mass > 20'),
    decay = cms.string('goodAODMuons@+ goodAODTrackCands@-')
)

process.dimuonsOneGlobalMuonOneTrackAOD = cms.EDFilter("CandViewRefSelector",
    src = cms.InputTag("dimuonsOneTrackAOD"),
    cut = cms.string('charge = 0 & daughter(0).isGlobalMuon = 1')
)

# dimuon filters
process.dimuonsFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("dimuonsAOD"),
    minNumber = cms.uint32(1)
)

process.dimuonsOneTrackFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("dimuonsOneTrackAOD"),
    minNumber = cms.uint32(1)
)

# Skim paths
process.EWK_dimuonsPath = cms.Path(
    process.EWK_MuHLTFilter *
    process.goodAODMuons *
    process.goodAODGlobalMuons *
    process.dimuonsAOD *
    process.dimuonsGlobalAOD *
    process.dimuonsOneStandAloneMuonAOD *
    process.dimuonsFilter
    )

process.EWK_dimuonsOneTrackPath = cms.Path(
    process.EWK_MuHLTFilter *
    process.goodAODMuons *
    process.goodAODGlobalMuons *
    process.trackCandsUnfiltered *
    process.goodAODTrackCands *
    process.dimuonsOneTrackAOD *
    process.dimuonsOneGlobalMuonOneTrackAOD *
    process.dimuonsOneTrackFilter
)


# Output module configuration
from Configuration.EventContent.EventContent_cff import *

EWK_dimuonsEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_trackCandsUnfiltered_*_*', 
        'keep *_goodAODTrackCands_*_*', 
        'keep *_goodAODMuons_*_*', 
        'keep *_dimuonsAOD_*_*', 
        'keep *_dimuonsGlobalAOD_*_*', 
        'keep *_dimuonsOneStandAloneMuonAOD_*_*', 
        'keep *_dimuonsOneTrackAOD_*_*', 
        'keep *_dimuonsOneGlobalMuonOneTrackAOD_*_*', 
        )
)

EWK_DimuonSkimEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
EWK_DimuonSkimEventContent.outputCommands.extend(AODEventContent.outputCommands)
EWK_DimuonSkimEventContent.outputCommands.extend(EWK_dimuonsEventContent.outputCommands)

EWK_DimuonSkimEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(
           'EWK_dimuonsPath',
           'EWK_dimuonsOneTrackPath')
    )
)

process.EWK_DimuonSkimOutputModule = cms.OutputModule("PoolOutputModule",
    EWK_DimuonSkimEventContent,
    EWK_DimuonSkimEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('EWKDimuonSkim'),
        dataTier = cms.untracked.string('USER')
   ),
   fileName = cms.untracked.string('file:testEWKDimuonSkim.root')
)

process.outpath = cms.EndPath(process.EWK_DimuonSkimOutputModule)


