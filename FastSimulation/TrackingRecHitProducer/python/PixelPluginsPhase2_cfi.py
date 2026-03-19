import FWCore.ParameterSet.Config as cms


pixelPluginsPhase2=cms.VPSet()


pixelPluginsPhase2.append(
    cms.PSet(
        select = cms.string("subdetId==BPX"),
        isBarrel = cms.bool(True),
        name   = cms.string("pixelBarrelSmearer"),
        type   = cms.string("PixelTemplateSmearerPlugin"),
        templateId                 = cms.int32(-1),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/3-18-25_barrel_v2.root'),
        BigPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/3-18-25_barrel_v2.root'),
        EdgePixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/3-18-25_barrel_v2.root'),
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
        templateId                 = cms.int32(-1),
        RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/3-18-25_forward_v2.root'),
        BigPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/3-18-25_forward_v2.root'),
        EdgePixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/3-18-25_forward_v2.root'),
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/fmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fysmear.root'),
    )
)

