import FWCore.ParameterSet.Config as cms


fastTrackerRecHits = cms.EDProducer("TrackingRecHitProducer",
    simHits = cms.InputTag("fastSimProducer","TrackerHits"),
    plugins=cms.VPSet()
)

fastTrackerRecHits.plugins.append(
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

fastTrackerRecHits.plugins.append(
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

trackerStripGaussianResolutions={
    "TIB": {
        1: cms.double(0.00195),
        2: cms.double(0.00191),
        3: cms.double(0.00325),
        4: cms.double(0.00323)
    },
    "TID": {
        1: cms.double(0.00262),
        2: cms.double(0.00354),
        3: cms.double(0.00391)
    },
    "TOB": {
        1: cms.double(0.00461),
        2: cms.double(0.00458),
        3: cms.double(0.00488),
        4: cms.double(0.00491),
        5: cms.double(0.00293),
        6: cms.double(0.00299)
    },
    "TEC": {
        1: cms.double(0.00262),
        2: cms.double(0.00354),
        3: cms.double(0.00391),
        4: cms.double(0.00346),
        5: cms.double(0.00378),
        6: cms.double(0.00508),
        7: cms.double(0.00422),
        8: cms.double(0.00434),
        9: cms.double(0.00432),
    }
}

for subdetId,trackerLayers in trackerStripGaussianResolutions.iteritems():
    for trackerLayer, resolutionX in trackerLayers.iteritems():
        pluginConfig = cms.PSet(
            name = cms.string(subdetId+str(trackerLayer)),
            type=cms.string("TrackingRecHitStripGSPlugin"),
            resolutionX=resolutionX,
            select=cms.string("(subdetId=="+subdetId+") && (layer=="+str(trackerLayer)+")"),
        )
        fastTrackerRecHits.plugins.append(pluginConfig)
