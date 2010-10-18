import FWCore.ParameterSet.Config as cms
process = cms.Process("TripletTest")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100))
process.source = cms.Source("PoolSource", fileNames =  cms.untracked.vstring(
    'file:data/cmssw20x/SingleMu_10m.root'))

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    destinations = cms.untracked.vstring('cout'),
    cout = cms.untracked.PSet( threshold = cms.untracked.string('INFO'))
)

#process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("RecoLocalTracker.Configuration.RecoLocalTracker_cff")
process.load("RecoTracker.Configuration.RecoTracker_cff")
process.load("RecoPixelVertexing.PixelTrackFitting.PixelTracks_cff")

from RecoLocalTracker.Configuration.RecoLocalTracker_cff import *
#process.siPixelClusters.src = cms.InputTag('simSiPixelDigis')


#from RecoPixelVertexing.PixelTriplets.PixelTripletHLTGenerator_cfi import *
from RecoPixelVertexing.PixelTrackFitting.PixelFitterByConformalMappingAndLine_cfi import *

from RecoPixelVertexing.PixelTrackFitting.PixelTracks_cfi import *
process.pixelTracks2 = pixelTracks.clone()
process.pixelTracks2.FitterPSet = PixelFitterByConformalMappingAndLine 

process.test = cms.EDAnalyzer("PixelTrackTest", TrackCollection = cms.string("pixelTracks2"))

process.p = cms.Path(pixeltrackerlocalreco+process.pixelTracks2+process.test)
