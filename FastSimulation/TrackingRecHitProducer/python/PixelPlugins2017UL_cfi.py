import FWCore.ParameterSet.Config as cms


pixelPlugins2016UL=cms.VPSet()

#=====================================================================================
#--- Phase 0 Pixel Barrel
#--- Layer 1
#
pixelPlugins2016UL.append(
    cms.PSet(
        select = cms.string("subdetId==BPX && pxbLayer==1"),
        isBarrel = cms.bool(True),
        name   = cms.string("pixelSmearerBarrelLayer1"),
        type   = cms.string("PixelTemplateSmearerPlugin"),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos09420.root'),
        BigPixelResolutionFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelEdge2017.root'),
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/bmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bysmear.root'),
    )
)

#=====================================================================================
#--- Phase 0 Pixel Barrel
#--- Layer 2
#
pixelPlugins2016UL.append(
    cms.PSet(
        select = cms.string("subdetId==BPX && pxbLayer==2"),
        isBarrel = cms.bool(True),
        name   = cms.string("pixelSmearerBarrelLayer2"),
        type   = cms.string("PixelTemplateSmearerPlugin"),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos05621.root'),
        BigPixelResolutionFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelEdge2017.root'),
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/bmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bysmear.root'),
    )
)

#=====================================================================================
#--- Phase 0 Pixel Barrel
#--- Layer 3
#
pixelPlugins2016UL.append(
    cms.PSet(
        select = cms.string("subdetId==BPX && pxbLayer==3"),
        isBarrel = cms.bool(True),
        name   = cms.string("pixelSmearerBarrelLayer3"),
        type   = cms.string("PixelTemplateSmearerPlugin"),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos07622.root'),
        BigPixelResolutionFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelEdge2017.root'),
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/bmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bysmear.root'),
    )
)


#=====================================================================================
#--- Phase 0 Pixel Forward
#--- Plaq1
#
pixelPlugins2016UL.append(
    cms.PSet(
        select=cms.string("subdetId==FPX && pxfModule==1"),
        isBarrel = cms.bool(False),
        name = cms.string("pixelSmearerForward"),
        type = cms.string("PixelTemplateSmearerPlugin"),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos05623.root'),
        BigPixelResolutionFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardEdge2017.root'),
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/fmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fysmear.root'),
    )
)

#--- Plaq2
pixelPlugins2016UL.append(
    cms.PSet(
        select=cms.string("subdetId==FPX && pxfModule==2"),
        isBarrel = cms.bool(False),
        name = cms.string("pixelSmearerForward"),
        type = cms.string("PixelTemplateSmearerPlugin"),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos07624.root'),
        BigPixelResolutionFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardEdge2017.root'),
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/fmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fysmear.root'),
    )
)

#--- Plaq3 and 4
pixelPlugins2016UL.append(
    cms.PSet(
        select=cms.string("subdetId==FPX && pxfModule>2"),
        isBarrel = cms.bool(False),
        name = cms.string("pixelSmearerForward"),
        type = cms.string("PixelTemplateSmearerPlugin"),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos06425.root'),
        BigPixelResolutionFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardEdge2017.root'),
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/fmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fysmear.root'),
    )
)

#---------------------------
pixelPlugins2017UL=cms.VPSet()


#=====================================================================================
#--- Phase 1 Pixel Barrel
#--- Layer 1
#
pixelPlugins2017UL.append(
    cms.PSet(
        select = cms.string("subdetId==BPX && pxbLayer==1"),
        isBarrel = cms.bool(True),
        name   = cms.string("pixelSmearerBarrelLayer1"),
        type   = cms.string("PixelTemplateSmearerPlugin"),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos02300.root'),
        BigPixelResolutionFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelEdge2017.root'),
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/bmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bysmear.root'),
    )
)

#
#--- Layer 2
#
pixelPlugins2017UL.append(
    cms.PSet(
        select = cms.string("subdetId==BPX && pxbLayer==2"),
        isBarrel = cms.bool(True),
        name   = cms.string("pixelSmearerBarrelLayer2"),
        type   = cms.string("PixelTemplateSmearerPlugin"),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos01410.root'),
        BigPixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelEdge2017.root'),
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/bmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bysmear.root'),
    )
)

#
#--- Layer 3
#
pixelPlugins2017UL.append(
    cms.PSet(
        select = cms.string("subdetId==BPX && pxbLayer==3"),
        isBarrel = cms.bool(True),
        name   = cms.string("pixelSmearerBarrelLayer3"),
        type   = cms.string("PixelTemplateSmearerPlugin"),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos01623.root'),
        BigPixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelEdge2017.root'),
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/bmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bysmear.root'),
    )
)

#
#--- Layer 4
#
pixelPlugins2017UL.append(
    cms.PSet(
        select = cms.string("subdetId==BPX && pxbLayer==4"),
        isBarrel = cms.bool(True),
        name   = cms.string("pixelSmearerBarrelLayer4"),
        type   = cms.string("PixelTemplateSmearerPlugin"),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos00333.root'),
        BigPixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelEdge2017.root'),
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/bmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bysmear.root'),
    )
)




#=====================================================================================
#--- Phase 1 Pixel Forward
#--- Ring 2, Panel 1
pixelPlugins2017UL.append(
    cms.PSet(
        select=cms.string("subdetId==FPX && pxfBlade>22 && pxfPanel==1"),  ## 1-56 (Ring 1 is 1-22, Ring 2 is 23-56)
        isBarrel = cms.bool(False),
        name = cms.string("pixelSmearerForwardRing2Panel1"),
        type = cms.string("PixelTemplateSmearerPlugin"),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos01740.root'),
        BigPixelResolutionFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardEdge2017.root'),
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/fmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fysmear.root'),
    )
)



#--- Ring 1, Panel 1
pixelPlugins2017UL.append(
    cms.PSet(
        select=cms.string("subdetId==FPX && pxfBlade<23 && pxfPanel==1"),  ## 1-56 (Ring 1 is 1-22, Ring 2 is 23-56)
        isBarrel = cms.bool(False),
        name = cms.string("pixelSmearerForwardRing1Panel1"),
        type = cms.string("PixelTemplateSmearerPlugin"),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos01850.root'),
        BigPixelResolutionFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardEdge2017.root'),
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/fmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fysmear.root'),
    )
)


#--- Ring 1, Panel 2
pixelPlugins2017UL.append(
    cms.PSet(
        select=cms.string("subdetId==FPX && pxfBlade<23 && pxfPanel==2"),  ## 1-56 (Ring 1 is 1-22, Ring 2 is 23-56)
        isBarrel = cms.bool(False),
        name = cms.string("pixelSmearerForwardRing1Panel2"),
        type = cms.string("PixelTemplateSmearerPlugin"),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos01860.root'),
        BigPixelResolutionFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardEdge2017.root'),
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/fmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fysmear.root'),
    )
)

#--- Ring 2, Panel 2
pixelPlugins2017UL.append(
    cms.PSet(
        select=cms.string("subdetId==FPX && pxfBlade>22 && pxfPanel==2"),  ## 1-56 (Ring 1 is 1-22, Ring 2 is 23-56)
        isBarrel = cms.bool(False),
        name = cms.string("pixelSmearerForwardRing2Panel2"),
        type = cms.string("PixelTemplateSmearerPlugin"),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos01770.root'),
        BigPixelResolutionFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardEdge2017.root'),
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/fmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fysmear.root'),
    )
)


#---------------------------
pixelPlugins2018UL=cms.VPSet()


#=====================================================================================
#--- Phase 1 Pixel Barrel
#--- Layer 1
#
pixelPlugins2018UL.append(
    cms.PSet(
        select = cms.string("subdetId==BPX && pxbLayer==1"),
        isBarrel = cms.bool(True),
        name   = cms.string("pixelSmearerBarrelLayer1"),
        type   = cms.string("PixelTemplateSmearerPlugin"),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos09409.root'),
        BigPixelResolutionFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelEdge2017.root'),
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/bmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bysmear.root'),
    )
)

#
#--- Layer 2
#
pixelPlugins2018UL.append(
    cms.PSet(
        select = cms.string("subdetId==BPX && pxbLayer==2"),
        isBarrel = cms.bool(True),
        name   = cms.string("pixelSmearerBarrelLayer2"),
        type   = cms.string("PixelTemplateSmearerPlugin"),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos08219.root'),
        BigPixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelEdge2017.root'),
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/bmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bysmear.root'),
    )
)

#
#--- Layer 3
#
pixelPlugins2018UL.append(
    cms.PSet(
        select = cms.string("subdetId==BPX && pxbLayer==3"),
        isBarrel = cms.bool(True),
        name   = cms.string("pixelSmearerBarrelLayer3"),
        type   = cms.string("PixelTemplateSmearerPlugin"),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos07939.root'),
        BigPixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelEdge2017.root'),
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/bmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bysmear.root'),
    )
)

#
#--- Layer 4
#
pixelPlugins2018UL.append(
    cms.PSet(
        select = cms.string("subdetId==BPX && pxbLayer==4"),
        isBarrel = cms.bool(True),
        name   = cms.string("pixelSmearerBarrelLayer4"),
        type   = cms.string("PixelTemplateSmearerPlugin"),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos07729.root'),
        BigPixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelEdge2017.root'),
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/bmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bysmear.root'),
    )
)




#=====================================================================================
#--- Phase 1 Pixel Forward
#--- Ring 2, Panel 1
pixelPlugins2018UL.append(
    cms.PSet(
        select=cms.string("subdetId==FPX && pxfBlade>22 && pxfPanel==1"),  ## 1-56 (Ring 1 is 1-22, Ring 2 is 23-56)
        isBarrel = cms.bool(False),
        name = cms.string("pixelSmearerForwardRing2Panel1"),
        type = cms.string("PixelTemplateSmearerPlugin"),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos06649.root'),
        BigPixelResolutionFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardEdge2017.root'),
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/fmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fysmear.root'),
    )
)



#--- Ring 1, Panel 1
pixelPlugins2018UL.append(
    cms.PSet(
        select=cms.string("subdetId==FPX && pxfBlade<23 && pxfPanel==1"),  ## 1-56 (Ring 1 is 1-22, Ring 2 is 23-56)
        isBarrel = cms.bool(False),
        name = cms.string("pixelSmearerForwardRing1Panel1"),
        type = cms.string("PixelTemplateSmearerPlugin"),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos06759.root'),
        BigPixelResolutionFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardEdge2017.root'),
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/fmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fysmear.root'),
    )
)


#--- Ring 1, Panel 2
pixelPlugins2018UL.append(
    cms.PSet(
        select=cms.string("subdetId==FPX && pxfBlade<23 && pxfPanel==2"),  ## 1-56 (Ring 1 is 1-22, Ring 2 is 23-56)
        isBarrel = cms.bool(False),
        name = cms.string("pixelSmearerForwardRing1Panel2"),
        type = cms.string("PixelTemplateSmearerPlugin"),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos06769.root'),
        BigPixelResolutionFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardEdge2017.root'),
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/fmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fysmear.root'),
    )
)

#--- Ring 2, Panel 2
pixelPlugins2018UL.append(
    cms.PSet(
        select=cms.string("subdetId==FPX && pxfBlade>22 && pxfPanel==2"),  ## 1-56 (Ring 1 is 1-22, Ring 2 is 23-56)
        isBarrel = cms.bool(False),
        name = cms.string("pixelSmearerForwardRing2Panel2"),
        type = cms.string("PixelTemplateSmearerPlugin"),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos06679.root'),
        BigPixelResolutionFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardBig2017.root'),
        EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/ForwardEdge2017.root'),
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/fmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fysmear.root'),
    )
)