import FWCore.ParameterSet.Config as cms


pixelPluginsPhase2=cms.VPSet()


pixelPluginsPhase2.append(
    cms.PSet(
        select = cms.string("subdetId==BPX"),
        isBarrel = cms.bool(True),
        name   = cms.string("pixelBarrelSmearer"),
        type   = cms.string("PixelTemplateSmearerPlugin"),
        templateId                 = cms.int32( 40 ),
        ## templateId                 = cms.int32( 292 ),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos58360_292.root'),
        BigPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos58360_292.root'),
        EdgePixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos58360_292.root'),
        ## BigPixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelEdge2017.root'),
        ## EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelEdge2017.root'),
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/bmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bysmear.root'),
    )
)

pixelPluginsPhase2.append(
    cms.PSet(
        select=cms.string("subdetId==FPX"),
        isBarrel = cms.bool(False),
        name = cms.string("pixelForwardSmearer"),
        type = cms.string("PixelTemplateSmearerPlugin"),
        templateId                 = cms.int32( 41 ),
        ## templateId                 = cms.int32( 291 ),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos49765_291.root'),
        BigPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos49765_291.root'),
        EdgePixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos49765_291.root'),
        ## BigPixelResolutionFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelEdge2017.root'),
        ## EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelEdge2017.root'),
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/fmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fysmear.root'),
    )
)

