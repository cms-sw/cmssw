import FWCore.ParameterSet.Config as cms

trackingRecHitProducer = cms.EDProducer("TrackingRecHitProducer",
     simHits = cms.InputTag("famosSimHits","TrackerHits"),
     ############ RunTrackingReco Example############ 
     plugins=cms.VPSet(
        cms.PSet(
            name = cms.string("pixelBarrelSmearer"),
            type=cms.string("PixelBarrelTemplateSmearerPlugin"),
            NewPixelBarrelResolutionFile1 = cms.string('FastSimulation/TrackingRecHitProducer/data/NewPixelResolutionBarrel38T.root'),
            NewPixelBarrelResolutionFile2 = cms.string('FastSimulation/TrackingRecHitProducer/data/NewPixelResolutionBarrelEdge38T.root'),
            NewPixelBarrelResolutionFile3 = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelBarrelResolution2014.root'),
            NewPixelForwardResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/NewPixelResolutionForward38T.root'),
            NewPixelForwardResolutionFile2 = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelForwardResolution2014.root'),
            UseCMSSWPixelParametrization = cms.bool(True),
            MergeHitsOn = cms.bool(False),
            probfilebarrel = cms.string('FastSimulation/TrackingRecHitProducer/data/bmergeprob.root'),
            probfileforward = cms.string('FastSimulation/TrackingRecHitProducer/data/fmergeprob.root'),
            pixelresxmergedbarrel = cms.string('FastSimulation/TrackingRecHitProducer/data/bxsmear.root'),
            pixelresxmergedforward = cms.string('FastSimulation/TrackingRecHitProducer/data/fxsmear.root'),
            pixelresymergedbarrel = cms.string('FastSimulation/TrackingRecHitProducer/data/bysmear.root'),
            pixelresymergedforward = cms.string('FastSimulation/TrackingRecHitProducer/data/fysmear.root'),
            templateIdBarrel = cms.int32( 40 ),
            templateIdForward  = cms.int32( 41 ),
            select=cms.string("subdetId==BPX"),
        ),
        
        cms.PSet(
            name = cms.string("BPXmonitor"),
            type=cms.string("TrackingRecHitMonitorPlugin"),
            xmax=cms.double(5.0),
            ymax=cms.double(5.0),
            select=cms.string("subdetId==BPX"),
         ),
        cms.PSet(
            name = cms.string("pixelForwardSmearer"),
            type=cms.string("PixelForwardTemplateSmearerPlugin"),
            NewPixelBarrelResolutionFile1 = cms.string('FastSimulation/TrackingRecHitProducer/data/NewPixelResolutionBarrel38T.root'),
            NewPixelBarrelResolutionFile2 = cms.string('FastSimulation/TrackingRecHitProducer/data/NewPixelResolutionBarrelEdge38T.root'),
            NewPixelBarrelResolutionFile3 = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelBarrelResolution2014.root'),
            NewPixelForwardResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/NewPixelResolutionForward38T.root'),
            NewPixelForwardResolutionFile2 = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelForwardResolution2014.root'),
            UseCMSSWPixelParametrization = cms.bool(True),
            MergeHitsOn = cms.bool(False),
            probfilebarrel = cms.string('FastSimulation/TrackingRecHitProducer/data/bmergeprob.root'),
            probfileforward = cms.string('FastSimulation/TrackingRecHitProducer/data/fmergeprob.root'),
            pixelresxmergedbarrel = cms.string('FastSimulation/TrackingRecHitProducer/data/bxsmear.root'),
            pixelresxmergedforward = cms.string('FastSimulation/TrackingRecHitProducer/data/fxsmear.root'),
            pixelresymergedbarrel = cms.string('FastSimulation/TrackingRecHitProducer/data/bysmear.root'),
            pixelresymergedforward = cms.string('FastSimulation/TrackingRecHitProducer/data/fysmear.root'),
            templateIdBarrel = cms.int32( 40 ),
            templateIdForward  = cms.int32( 41 ),
            select=cms.string("subdetId==FPX"),
        ),
        
        cms.PSet(
            name = cms.string("FPXmonitor"),
            type=cms.string("TrackingRecHitMonitorPlugin"),
            xmax=cms.double(5.0),
            ymax=cms.double(5.0),
            select=cms.string("subdetId==FPX"),
         )
     )
                                           
)

