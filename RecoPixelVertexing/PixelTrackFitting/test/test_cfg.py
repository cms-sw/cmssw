import FWCore.ParameterSet.Config as cms
process = cms.Process("Analysis")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000))
process.source = cms.Source("PoolSource", fileNames =  cms.untracked.vstring( 
'/store/relval/CMSSW_3_7_0_pre4/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V3-v1/0022/FEC7E481-A85D-DF11-8A78-003048678B04.root'
))

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'MC_37Y_V3::All'

process.MessageLogger = cms.Service("MessageLogger",
    #debugModules = cms.untracked.vstring('pixelVertices'),
    #debugModules = cms.untracked.vstring('pixelTracks'),
    #debugModules = cms.untracked.vstring('*'),
    debugModules = cms.untracked.vstring(''),
    destinations = cms.untracked.vstring('cout'),
    cout = cms.untracked.PSet( threshold = cms.untracked.string('INFO'))
)

process.load("RecoTracker.Configuration.RecoTracker_cff")
from RecoTracker.Configuration.RecoTracker_cff import *
process.load('RecoLocalTracker/Configuration/RecoLocalTracker_cff')
process.load("RecoPixelVertexing.PixelTrackFitting.PixelTracks_cff")
from RecoPixelVertexing.PixelTrackFitting.PixelTracks_cff import *
from RecoPixelVertexing.PixelTrackFitting.PixelFitterByConformalMappingAndLine_cfi import *


BBlock = cms.PSet(
  ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
  RegionPSet = cms.PSet(
    precise = cms.bool(True),
    nSigmaZ = cms.double(3.0),
    originRadius = cms.double(0.2),
    ptMin = cms.double(0.8),
    beamSpot = cms.InputTag("offlineBeamSpot")
  )
)


PixelTripletHLTGenerator = cms.PSet (
  ComponentName = cms.string('PixelTripletHLTGenerator'),
  useFixedPreFiltering = cms.bool(True),
  phiPreFiltering = cms.double(0.3),
  useBending = cms.bool(True),
  extraHitRPhitolerance = cms.double(0.032),
  useMultScattering = cms.bool(True),
  extraHitRZtolerance = cms.double(0.037),
  maxTriplets = cms.uint32(10000),
  maxElement = cms.uint32(10000)
)


FitterPSet2 = cms.PSet(
  ComponentName = cms.string('PixelFitterByConformalMappingAndLine'),
#  fixImpactParameter = cms.double(0.),
#  ComponentName = cms.string('PixelFitterByHelixProjections'),
  TTRHBuilder   = cms.string('TTRHBuilderWithoutAngle4PixelTriplets')
)

FitterPSet3 = cms.PSet(
  ComponentName = cms.string('PixelFitterByConformalMappingAndLine'),
  fixImpactParameter = cms.double(0.),
#  ComponentName = cms.string('PixelFitterByHelixProjections'),
  TTRHBuilder   = cms.string('TTRHBuilderWithoutAngle4PixelTriplets')
)

process.pixelTracks2 = pixelTracks.clone()
process.pixelTracks2.RegionFactoryPSet= cms.PSet( BBlock )
process.pixelTracks2.FilterPSet.ComponentName = cms.string('none')
process.pixelTracks2.OrderedHitsFactoryPSet.GeneratorPSet = cms.PSet ( PixelTripletHLTGenerator )
process.pixelTracks2.FitterPSet = cms.PSet(FitterPSet2)

process.pixelTracks3 = process.pixelTracks2.clone()
process.pixelTracks3.FitterPSet = cms.PSet(FitterPSet3)

process.test1 = cms.EDAnalyzer("PixelTrackTest", TrackCollection = cms.string("pixelTracks"))
process.test2 = cms.EDAnalyzer("PixelTrackTest", TrackCollection = cms.string("pixelTracks2"))
process.test3 = cms.EDAnalyzer("PixelTrackTest", TrackCollection = cms.string("pixelTracks3"))

process.p=cms.Path(process.siPixelDigis*process.pixeltrackerlocalreco*process.offlineBeamSpot*process.pixelTracks*process.pixelTracks2*process.pixelTracks3*process.test1*process.test2*process.test3)

