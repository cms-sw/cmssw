import FWCore.ParameterSet.Config as cms
process = cms.Process("TripletTest")

import FWCore.ParameterSet.Config as cms

process = cms.Process("TKSEEDING")

# message logger
#process.MessageLogger = cms.Service("MessageLogger",
#     default = cms.untracked.PSet( limit = cms.untracked.int32(10) )
#)

#Adding SimpleMemoryCheck service:
process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
                                   ignoreTotal=cms.untracked.int32(1),
                                   oncePerEventMode=cms.untracked.bool(False)
)


# source
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( ['file:step3.root' ])
secFiles.extend( ['file:step2.root'] )

process.source = source
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

### conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'STARTUP3X_V14::All'

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')

### standard includes
process.load('Configuration/StandardSequences/Services_cff')
process.load('Configuration.StandardSequences.Geometry_cff')
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")


process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    destinations = cms.untracked.vstring('cout'),
    cout = cms.untracked.PSet( threshold = cms.untracked.string('INFO'))
)

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

process.p = cms.Path(process.siPixelRecHits+process.PixelLayerTriplets+process.triplets)
