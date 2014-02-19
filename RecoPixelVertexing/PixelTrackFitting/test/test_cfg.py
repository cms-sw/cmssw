import FWCore.ParameterSet.Config as cms
process = cms.Process("Analysis")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000))
process.source = cms.Source("PoolSource", fileNames =  cms.untracked.vstring( 
'file:data/DoubleMu_3_00.root'
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
process.GlobalTag.globaltag = 'DESIGN_36_V3::All'

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

GBlock= cms.PSet(
  ComponentName = cms.string('GlobalRegionProducer'),
  RegionPSet = cms.PSet(
     precise = cms.bool(True),
     ptMin = cms.double(0.875),
     originHalfLength = cms.double(15.9),
     originRadius = cms.double(0.2),
     originXPos = cms.double(0.),
     originYPos = cms.double(0.),
     originZPos = cms.double(0.)
#     originXPos = cms.double(0.2),
#     originYPos = cms.double(0.4),
#     originZPos = cms.double(-2.4)
      
  )
)


process.pixelTracks2 = pixelTracks.clone()
process.pixelTracks2.RegionFactoryPSet= cms.PSet( GBlock )
process.pixelTracks2.FilterPSet.ComponentName = cms.string('none')

process.test = cms.EDAnalyzer("PixelTrackTest", TrackCollection = cms.string("pixelTracks2"))

process.p=cms.Path(process.siPixelDigis*process.pixeltrackerlocalreco*process.offlineBeamSpot*process.PixelLayerTriplets*process.pixelTracks2*process.test)

