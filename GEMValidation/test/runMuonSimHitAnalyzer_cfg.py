import FWCore.ParameterSet.Config as cms

process = cms.Process("MUONSIMANA")

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
from GEMCode.GEMValidation.simTrackMatching_cfi import SimTrackMatching
process.MuonSimHitAnalyzer = cms.EDAnalyzer("MuonSimHitAnalyzer",
    simTrackMatching = SimTrackMatching
)
process.MuonSimHitAnalyzer.simTrackMatching.gemStripDigi.input = ""
process.MuonSimHitAnalyzer.simTrackMatching.gemPadDigi.input = ""
process.MuonSimHitAnalyzer.simTrackMatching.gemCoPadDigi.input = ""
process.MuonSimHitAnalyzer.simTrackMatching.cscStripDigi.input = ""
process.MuonSimHitAnalyzer.simTrackMatching.cscWireDigi.input = ""
process.MuonSimHitAnalyzer.simTrackMatching.cscCLCT.input = ""
process.MuonSimHitAnalyzer.simTrackMatching.cscALCT.input = ""
process.MuonSimHitAnalyzer.simTrackMatching.cscLCT.input = ""
process.MuonSimHitAnalyzer.simTrackMatching.cscMPLCT.input = ""
process.MuonSimHitAnalyzer.simTrackMatching.gemRecHit.input = ""
process.MuonSimHitAnalyzer.simTrackMatching.tfTrack.input = ""
process.MuonSimHitAnalyzer.simTrackMatching.tfCand.input = ""
process.MuonSimHitAnalyzer.simTrackMatching.gmtCand.input = ""
process.MuonSimHitAnalyzer.simTrackMatching.l1Extra.input = ""

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring("file:out_sim.root")                            
)

process.TFileService = cms.Service("TFileService",
  fileName = cms.string("gem_sh_ana.root")
)

process.p = cms.Path(process.MuonSimHitAnalyzer)

