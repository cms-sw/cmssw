import FWCore.ParameterSet.Config as cms

process = cms.Process("GEMSIMANA")

## Standard sequence
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
#process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023HGCalReco_cff')
process.load('Configuration.Geometry.GeometryExtended2023HGCal_cff')
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
process.load('GEMCode.GEMValidation.GEMSimHitAnalyzer_cfi')
process.GEMSimHitAnalyzer.simTrackMatching.gemDigi.input = ""
process.GEMSimHitAnalyzer.simTrackMatching.gemPadDigi.input = ""
process.GEMSimHitAnalyzer.simTrackMatching.gemCoPadDigi.input = ""
process.GEMSimHitAnalyzer.simTrackMatching.cscComparatorDigi.input = ""
process.GEMSimHitAnalyzer.simTrackMatching.cscWireDigi.input = ""
process.GEMSimHitAnalyzer.simTrackMatching.cscCLCT.input = ""
process.GEMSimHitAnalyzer.simTrackMatching.cscALCT.input = ""
process.GEMSimHitAnalyzer.simTrackMatching.cscLCT.input = ""
process.GEMSimHitAnalyzer.simTrackMatching.gemRecHit.input = ""
 
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring("file:out_sim.root")                            
)

process.TFileService = cms.Service("TFileService",
  fileName = cms.string("gem_sh_ana.root")
)

process.p = cms.Path(process.GEMSimHitAnalyzer)

