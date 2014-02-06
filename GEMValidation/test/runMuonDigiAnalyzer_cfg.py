import FWCore.ParameterSet.Config as cms

process = cms.Process("GEMDIGIANA")

## Standard sequence
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

## TrackingComponentsRecord required for matchers
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi')
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi')

## global tag for 2019 upgrade studies
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')

# the analyzer configuration
process.load('GEMCode.GEMValidation.GEMDigiAnalyzer_cfi')
process.GEMDigiAnalyzer.simTrackMatching.cscComparatorDigiInput = ""
process.GEMDigiAnalyzer.simTrackMatching.cscWireDigiInput = ""
process.GEMDigiAnalyzer.simTrackMatching.cscCLCTInput = ""
process.GEMDigiAnalyzer.simTrackMatching.cscALCTInput = ""
process.GEMDigiAnalyzer.simTrackMatching.cscLCTInput = ""
process.GEMDigiAnalyzer.simTrackMatching.gemRecHitInput = ""

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring("file:out_digi.root")
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string("gem_digi_ana.root")
)

process.p = cms.Path(process.GEMDigiAnalyzer)

