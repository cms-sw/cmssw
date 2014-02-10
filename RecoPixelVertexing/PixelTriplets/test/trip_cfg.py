import FWCore.ParameterSet.Config as cms
process = cms.Process("TripletTest")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000))
process.source = cms.Source("PoolSource", fileNames =  cms.untracked.vstring(
 'file:data/cmssw20x/ttbar.root'
#  'file:data/cmssw20x/SingleMu_10m.root'
))

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

from RecoLocalTracker.Configuration.RecoLocalTracker_cff import *
process.siPixelClusters.src = cms.InputTag('simSiPixelDigis')

from RecoPixelVertexing.PixelTriplets.PixelTripletHLTGenerator_cfi import *
from RecoPixelVertexing.PixelTriplets.PixelTripletLargeTipGenerator_cfi import *
from RecoPixelVertexing.PixelTrackFitting.PixelTracks_cfi import *
from RecoTracker.TkTrackingRegions.GlobalTrackingRegion_cfi import *

process.triplets = cms.EDAnalyzer("HitTripletProducer",
  OrderedHitsFactoryPSet = cms.PSet(
    ComponentName = cms.string("StandardHitTripletGenerator"),
    SeedingLayers = cms.InputTag("PixelLayerTriplets"),
    GeneratorPSet = cms.PSet( PixelTripletHLTGenerator )
#    GeneratorPSet = cms.PSet( PixelTripletLargeTipGenerator )
  ),
    RegionFactoryPSet = cms.PSet(
        RegionPSetBlock,
        ComponentName = cms.string('GlobalRegionProducer')
  )

)
#process.triplets.OrderedHitsFactoryPSet.GeneratorPSet.useFixedPreFiltering = cms.bool(True)
#process.triplets.RegionFactoryPSet.RegionPSet.ptMin = cms.double(1000.00)
#process.triplets.RegionFactoryPSet.RegionPSet.originRadius = cms.double(0.001)
#process.triplets.RegionFactoryPSet.RegionPSet.originHalfLength = cms.double(0.0001)

process.p = cms.Path(pixeltrackerlocalreco+process.PixelLayerTriplets+process.triplets)
