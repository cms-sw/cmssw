import FWCore.ParameterSet.Config as cms

process = cms.Process("EWKWZSkim")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.options.FailPath = cms.untracked.vstring('ProductNotFound')

# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
    'file:/scratch2/users/fabozzi/spring10/wmn/24BF0D12-DF46-DF11-BA71-001D0968F2F6.root'
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('START36_V8::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

# HLT filter
import HLTrigger.HLTfilters.hltHighLevel_cfi
process.EWK_MuHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
process.EWK_MuHLTFilter.HLTPaths = ["HLT_Mu9"]

# Muon candidates filters 
process.goodMuons = cms.EDFilter("MuonSelector",
  src = cms.InputTag("muons"),
  cut = cms.string('pt > 20 && abs(eta)<2.4 && isGlobalMuon = 1 && isTrackerMuon = 1 && isolationR03().sumPt<3.0'),
  filter = cms.bool(True)                                
)

# dxy filter on good muons
process.dxyFilteredMuons = cms.EDFilter("MuonSelector",
  src = cms.InputTag("goodMuons"),
  cut = cms.string('abs(innerTrack().dxy)<1.0'),
  filter = cms.bool(True)                                
)

# Z->mumu candidates
process.dimuons = cms.EDProducer("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(True),
    cut = cms.string('mass > 60'),
    decay = cms.string("dxyFilteredMuons@+ dxyFilteredMuons@-")
)

# Z filters
process.dimuonsFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("dimuons"),
    minNumber = cms.uint32(1)
)

# WMuNu candidates
process.load("ElectroWeakAnalysis.WMuNu.wmunusProducer_cfi")
# WMuNu candidates selectors
process.load("ElectroWeakAnalysis.WMuNu.WMuNuSelection_cff")

process.seltcMet.JetTag = cms.untracked.InputTag("ak5CaloJets")
process.seltcMet.TrigTag = cms.untracked.InputTag("TriggerResults::HLT")
process.seltcMet.IsCombinedIso = cms.untracked.bool(True)
process.seltcMet.IsoCut03 = cms.untracked.double(0.15)

process.selpfMet.JetTag = cms.untracked.InputTag("ak5CaloJets")
process.selpfMet.TrigTag = cms.untracked.InputTag("TriggerResults::HLT")
process.selpfMet.IsCombinedIso = cms.untracked.bool(True)
process.selpfMet.IsoCut03 = cms.untracked.double(0.15)

# Skim paths
process.EWK_dimuonsPath = cms.Path(
    process.EWK_MuHLTFilter *
    process.goodMuons *
    process.dxyFilteredMuons *
    process.dimuons *
    process.dimuonsFilter
    )

process.EWK_tcMetWMuNusPath = cms.Path(
    process.EWK_MuHLTFilter *
    process.tcMetWMuNus *
    process.seltcMet
)

process.EWK_pfMetWMuNusPath = cms.Path(
    process.EWK_MuHLTFilter *
    process.pfMetWMuNus *
    process.selpfMet
)

# Output module configuration
from Configuration.EventContent.EventContent_cff import *
EWK_WZSkimEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
EWK_WZSkimEventContent.outputCommands.extend(FEVTEventContent.outputCommands)

EWK_WZSkimEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(
           'EWK_dimuonsPath',
           'EWK_tcMetWMuNusPath',
           'EWK_pfMetWMuNusPath')
    )
)

process.EWK_WZSkimOutputModule = cms.OutputModule("PoolOutputModule",
    EWK_WZSkimEventContent,
    EWK_WZSkimEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('EWKWZSkim'),
        dataTier = cms.untracked.string('USER')
   ),
   fileName = cms.untracked.string('EWK_WZSkim_SD_Mu.root')
)

process.outpath = cms.EndPath(process.EWK_WZSkimOutputModule)


