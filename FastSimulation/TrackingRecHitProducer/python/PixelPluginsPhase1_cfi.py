import FWCore.ParameterSet.Config as cms

pixelPluginsPhase1=cms.VPSet()


#=====================================================================================
#--- Phase 1 Pixel Barrel
#
#    Layer    Template       Cluster file                 Resolution histograms
#    -----------------------------------------------------------------------------
#    BPL1      2403     template_events_d63207.out.gz   pixel_histos63207_2403.root
#
#--- Layer 1
#
pixelPluginsPhase1.append(
    cms.PSet(
        select = cms.string("subdetId==BPX && pxbLayer==1"),
        isBarrel = cms.bool(True),
        name   = cms.string("pixelSmearerBarrelLayer1"),
        type   = cms.string("PixelTemplateSmearerPlugin"),
        # templateId                 = cms.int32( 2403 ),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos63207_2403_6.root'),
        BigPixelResolutionFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelEdge2017.root'),
        #
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/bmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bysmear.root'),
    )
)

#
#--- Layer 2
#    BPL2      2413     template_events_d63507.out.gz   pixel_histos63507_2413.root
#
pixelPluginsPhase1.append(
    cms.PSet(
        select = cms.string("subdetId==BPX && pxbLayer==2"),
        isBarrel = cms.bool(True),
        name   = cms.string("pixelSmearerBarrelLayer2"),
        type   = cms.string("PixelTemplateSmearerPlugin"),
        # templateId                 = cms.int32( 2413 ),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos63507_2413_6.root'),
        BigPixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelEdge2017.root'),
        #
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/bmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bysmear.root'),
    )
)

#
#--- Layer 3
#    BPL3      2423     template_events_d63807.out.gz   pixel_histos63807_2423.root
#
pixelPluginsPhase1.append(
    cms.PSet(
        select = cms.string("subdetId==BPX && pxbLayer==3"),
        isBarrel = cms.bool(True),
        name   = cms.string("pixelSmearerBarrelLayer3"),
        type   = cms.string("PixelTemplateSmearerPlugin"),
        # templateId                 = cms.int32( 2413 ),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos63807_2423_6.root'),
        BigPixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelEdge2017.root'),
        #
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/bmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bysmear.root'),
    )
)


#
#--- Layer 4
#    BPL4      2433     template_events_d63807.out.gz   pixel_histos64107_2433.root
#
pixelPluginsPhase1.append(
    cms.PSet(
        select = cms.string("subdetId==BPX && pxbLayer==4"),
        isBarrel = cms.bool(True),
        name   = cms.string("pixelSmearerBarrelLayer4"),
        type   = cms.string("PixelTemplateSmearerPlugin"),
        # templateId                 = cms.int32( 2413 ),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos64107_2433_6.root'),
        BigPixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelEdge2017.root'),
        #
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/bmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bysmear.root'),
    )
)




#=====================================================================================
#--- Phase 1 Pixel Forward
#
#    Panel    Template       Cluster file                 Resolution histograms
#    -----------------------------------------------------------------------------
#    FPR2P1    2443     template_events_d64237.out.gz     pixel_histos64237_2443.root
#
#--- Ring 2, Panel 1
pixelPluginsPhase1.append(
    cms.PSet(
        select=cms.string("subdetId==FPX && pxfBlade>22 && pxfPanel==1"),  ## 1-56 (Ring 1 is 1-22, Ring 2 is 23-56)
        isBarrel = cms.bool(False),
        name = cms.string("pixelSmearerForwardRing2Panel1"),
        type = cms.string("PixelTemplateSmearerPlugin"),
        # templateId                 = cms.int32( 2443 ),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos64237_2443_6.root'),
        BigPixelResolutionFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardEdge2017.root'),
        #
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/fmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fysmear.root'),
    )
)



#--- Ring 1, Panel 1
#    FPR1P1    2453     template_events_d64367.out.gz     pixel_histos64367_2453.root
pixelPluginsPhase1.append(
    cms.PSet(
        select=cms.string("subdetId==FPX && pxfBlade<23 && pxfPanel==1"),  ## 1-56 (Ring 1 is 1-22, Ring 2 is 23-56)
        isBarrel = cms.bool(False),
        name = cms.string("pixelSmearerForwardRing1Panel1"),
        type = cms.string("PixelTemplateSmearerPlugin"),
        # templateId                 = cms.int32( 2453 ),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos64367_2453_6.root'),
        BigPixelResolutionFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardEdge2017.root'),
        #
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/fmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fysmear.root'),
    )
)


#--- Ring 1, Panel 2
#    FPR1P2    2463     template_events_d64497.out.gz     pixel_histos64497_2463.root
pixelPluginsPhase1.append(
    cms.PSet(
        select=cms.string("subdetId==FPX && pxfBlade<23 && pxfPanel==2"),  ## 1-56 (Ring 1 is 1-22, Ring 2 is 23-56)
        isBarrel = cms.bool(False),
        name = cms.string("pixelSmearerForwardRing1Panel2"),
        type = cms.string("PixelTemplateSmearerPlugin"),
        # templateId                 = cms.int32( 2463 ),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos64497_2463_6.root'),
        BigPixelResolutionFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardEdge2017.root'),
        #
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/fmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fysmear.root'),
    )
)

#--- Ring 2, Panel 2
#    FPR2P2    2473     template_events_d64627.out.gz     pixel_histos64627_2473.root
pixelPluginsPhase1.append(
    cms.PSet(
        select=cms.string("subdetId==FPX && pxfBlade>22 && pxfPanel==2"),  ## 1-56 (Ring 1 is 1-22, Ring 2 is 23-56)
        isBarrel = cms.bool(False),
        name = cms.string("pixelSmearerForwardRing2Panel2"),
        type = cms.string("PixelTemplateSmearerPlugin"),
        # templateId                 = cms.int32( 2473 ),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos64627_2473_6.root'),
        BigPixelResolutionFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardEdge2017.root'),
        #
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/fmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fysmear.root'),
    )
)

