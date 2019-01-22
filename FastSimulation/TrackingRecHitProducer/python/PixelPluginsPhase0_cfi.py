import FWCore.ParameterSet.Config as cms


pixelPluginsPhase0=cms.VPSet()

#=====================================================================================
#--- Phase 0 Pixel Barrel
#
#    Layer    Template       Cluster file                 Resolution histograms
#    -----------------------------------------------------------------------------
#      *      5611       template_events_d39921.out.gz    pixel_histos39921_5611_6.root
#
#
pixelPluginsPhase0.append(
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
pixelPluginsPhase0.append(
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


