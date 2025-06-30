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
        #RegularPixelResolutionFile = cms.string('/eos/cms/store/group/offcomp-sim/FastSimPhase2/3-18-25_barrel.root'),
        #BigPixelResolutionFile = cms.string('/eos/cms/store/group/offcomp-sim/FastSimPhase2/3-18-25_barrel.root'),
        #EdgePixelResolutionFile = cms.string('/eos/cms/store/group/offcomp-sim/FastSimPhase2/3-18-25_barrel.root'),        
        
        RegularPixelResolutionFile = cms.string('pixelhistos/3-18-25_barrel_v2.root'),
        BigPixelResolutionFile = cms.string('pixelhistos/3-18-25_barrel_v2.root'),
        EdgePixelResolutionFile = cms.string('pixelhistos/3-18-25_barrel_v2.root'),        
        
        #phase1
        #RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos39921_5611_6.root'),
        #BigPixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelBig2017.root'),
        #EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelEdge2017.root'),

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
#        RegularPixelResolutionFile = cms.string('/eos/cms/store/group/offcomp-sim/FastSimPhase2/3-18-25_forward.root'),
#        BigPixelResolutionFile = cms.string('/eos/cms/store/group/offcomp-sim/FastSimPhase2/3-18-25_forward.root'),
#        EdgePixelResolutionFile = cms.string('/eos/cms/store/group/offcomp-sim/FastSimPhase2/3-18-25_forward.root'),
        RegularPixelResolutionFile = cms.string('pixelhistos/3-18-25_forward_v2.root'),
        BigPixelResolutionFile = cms.string('pixelhistos/3-18-25_forward_v2.root'),
        EdgePixelResolutionFile = cms.string('pixelhistos/3-18-25_forward_v2.root'),
        #old files
        #RegularPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos49765_291.root'),
        #BigPixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos49765_291.root'),
        #EdgePixelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/pixel_histos49765_291.root'),
        ## BigPixelResolutionFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelEdge2017.root'),
        ## EdgePixelResolutionFile    = cms.string('FastSimulation/TrackingRecHitProducer/data/BarrelEdge2017.root'),
        MergeHitsOn                = cms.bool(False),
        MergingProbabilityFile     = cms.string('FastSimulation/TrackingRecHitProducer/data/fmergeprob.root'),
        MergedPixelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fxsmear.root'),
        MergedPixelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fysmear.root'),
    )
)

