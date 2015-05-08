import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelTriplets.PixelTripletHLTGenerator_cfi import *
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
from RecoPixelVertexing.PixelTrackFitting.PixelFitterByConformalMappingAndLine_cfi import *
from RecoHI.HiTracking.HIPixelTrackFilter_cfi import *
from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *

hiConformalPixelTracks = cms.EDProducer("PixelTrackProducer",
                                        
                                        #passLabel  = cms.string('Pixel triplet low-pt tracks with vertex constraint'),
                                        
                                        # Region
                                        RegionFactoryPSet = cms.PSet(
    ComponentName = cms.string("GlobalTrackingRegionWithVerticesProducer"),
    RegionPSet = cms.PSet(
    HiLowPtTrackingRegionWithVertexBlock
    )
    ),
                                        
                                        # Ordered Hits
                                        OrderedHitsFactoryPSet = cms.PSet( 
    ComponentName = cms.string( "StandardHitTripletGenerator" ),
    SeedingLayers = cms.InputTag( "PixelLayerTriplets" ),
    GeneratorPSet = cms.PSet( 
    PixelTripletHLTGenerator
    )
    ),
                                        
                                        # Fitter
                                        FitterPSet = cms.PSet( PixelFitterByConformalMappingAndLine 
                                                               ),
                                        
                                        # Filter
                                        useFilterWithES = cms.bool( True ),
                                        FilterPSet = cms.PSet( 
    HiConformalPixelFilterBlock 
    ),
                                        
                                        # Cleaner
                                        CleanerPSet = cms.PSet(  
    ComponentName = cms.string( "TrackCleaner" )
    )
                                        )

# increase threshold for triplets in generation step (default: 10000)
hiConformalPixelTracks.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = 5000000

hiConformalPixelTracks.FitterPSet.fixImpactParameter = cms.double(0.0)
hiConformalPixelTracks.FitterPSet.TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets')
