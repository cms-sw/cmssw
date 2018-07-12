import FWCore.ParameterSet.Config as cms


fastTrackerRecHits = cms.EDProducer("TrackingRecHitProducer",
    simHits = cms.InputTag("fastSimProducer","TrackerHits"),
    plugins=cms.VPSet()
)

#=====================================================================================
#--- Phase 0 Pixel Barrel
#
#    Layer    Template       Cluster file                 Resolution histograms
#    -----------------------------------------------------------------------------
#      *      5611       template_events_d39921.out.gz    pixel_histos39921_5611_6.root
#
#
fastTrackerRecHits.plugins.append(
    cms.PSet(
        select = cms.string("subdetId==BPX"),
        isBarrel = cms.bool(True),
        name   = cms.string("pixelSmearerBarrel"),
        type   = cms.string("PixelTemplateSmearerPlugin"),
        # templateId                 = cms.int32( 5611 ),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos39921_5611_6.root'),
        BigPixelResolutionFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelEdge2017.root'),
        #
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/bmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bysmear.root'),
    )
)


#=====================================================================================
#--- Phase 0 Pixel Forward
#
#    Panel    Template       Cluster file                 Resolution histograms
#    -----------------------------------------------------------------------------
#       *       6415      template_events_d40722.out.gz   pixel_histos40722_6415_6.root
#
#
fastTrackerRecHits.plugins.append(
    cms.PSet(
        select=cms.string("subdetId==FPX"),
        isBarrel = cms.bool(False),
        name = cms.string("pixelSmearerForward"),
        type = cms.string("PixelTemplateSmearerPlugin"),
        # templateId                 = cms.int32( 6415 ),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos40722_6415_6.root'),
        BigPixelResolutionFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardEdge2017.root'),
        #
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
