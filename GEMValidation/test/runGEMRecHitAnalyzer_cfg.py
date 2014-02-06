mport FWCore.ParameterSet.Config as cms

process = cms.Process("GEMRECOANA")

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
from GEMCode.GEMValidation.simTrackMatching_cfi import SimTrackMatching
GEMRecHitAnalyzer = cms.EDAnalyzer("GEMRecHitAnalyzer",
    simTrackMatching = SimTrackMatching
)
process.GEMRecHitAnalyzer.simTrackMatching.gemStripDigi.input = ""
process.GEMRecHitAnalyzer.simTrackMatching.gemPadDigi.input = ""
process.GEMRecHitAnalyzer.simTrackMatching.gemCoPadDigi.input = ""
process.GEMRecHitAnalyzer.simTrackMatching.cscStripDigi.input = ""
process.GEMRecHitAnalyzer.simTrackMatching.cscWireDigi.input = ""
process.GEMRecHitAnalyzer.simTrackMatching.cscCLCT.input = ""
process.GEMRecHitAnalyzer.simTrackMatching.cscALCT.input = ""
process.GEMRecHitAnalyzer.simTrackMatching.cscLCT.input = ""
process.GEMRecHitAnalyzer.simTrackMatching.cscMPLCT.input = ""
process.GEMRecHitAnalyzer.simTrackMatching.tfTrack.input = ""
process.GEMRecHitAnalyzer.simTrackMatching.tfCand.input = ""
process.GEMRecHitAnalyzer.simTrackMatching.gmtCand.input = ""
process.GEMRecHitAnalyzer.simTrackMatching.l1Extra.input = ""

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:out_rechit.root')
)

process.TFileService = cms.Service("TFileService",
  fileName = cms.string("gem_localrec_ana.root")
)

process.p = cms.Path(process.GEMRecHitAnalyzer)
