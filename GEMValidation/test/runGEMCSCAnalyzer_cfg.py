import FWCore.ParameterSet.Config as cms

process = cms.Process("GEMCSCANA")

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

process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2019_cff')

## GEM geometry customization
from Geometry.GEMGeometry.gemGeometryCustoms import custom_GE11_6partitions_v1
process = custom_GE11_6partitions_v1(process)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('out_digi.root')
)

## input
from GEMCode.SimMuL1.GEMCSCTriggerSamplesLib import files
from GEMCode.GEMValidation.InputFileHelpers import useInputDir
process = useInputDir(process, files['_gem98_pt2-50_PU0_pt0_new'], False)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string("gem-csc_stub_ana.root")
)

##output file name
theOutputFile = "out_GEMCSC_Ana.root"
process.TFileService.fileName = theOutputFile
print "OutputFiles: ", theOutputFile

## global tag for 2019 upgrade studies
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')

# the analyzer configuration
from GEMCode.GEMValidation.simTrackMatching_cfi import SimTrackMatching
process.GEMCSCAnalyzer = cms.EDAnalyzer("GEMCSCAnalyzer",
    verbose = cms.untracked.int32(0),
    stationsToUse = cms.vint32(1,2,3,4),
    simTrackMatching = simTrackMatching
)
matching = process.GEMCSCAnalyzer.simTrackMatching
matching.simTrack.minPt = 1.5
matching.cscSimHit.minNHitsChamber = 3
matching.cscStripDigi.minNHitsChamber = 3
matching.cscWireDigi.minNHitsChamber = 3
matching.cscCLCT.minNHitsChamber = 3
matching.cscALCT.minNHitsChamber = 3
matching.cscLCT.minNHitsChamber = 3
matching.cscMPLCT.minNHitsChamber = 3
matching.gemRecHit.input = ""
matching.tfTrack.input = ""
matching.tfCand.input = ""
matching.gmtCand.input = ""
matching.l1Extra.input = ""

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.p = cms.Path(process.GEMCSCAnalyzer)

