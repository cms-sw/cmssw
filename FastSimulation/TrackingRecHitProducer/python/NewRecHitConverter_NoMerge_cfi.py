import FWCore.ParameterSet.Config as cms


trackingRecHitProducerNoMerge = cms.EDProducer("TrackingRecHitProducer",
     simHits = cms.InputTag("famosSimHits","TrackerHits"),
     plugins=cms.VPSet()                                      
)

trackingRecHitProducerNoMerge.plugins.append(
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
    )
)

trackingRecHitProducerNoMerge.plugins.append(
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
    }  
}

for subdetId,trackerLayers in trackerStripGaussianResolutions.iteritems():
    for trackerLayer, resolutionX in trackerLayers.iteritems():
        pluginConfig = cms.PSet(
            name = cms.string(subdetId+str(trackerLayer)),
            type=cms.string("TrackingRecHitStripGSSmearingPlugin"),
            resolutionX=resolutionX,
            select=cms.string("(subdetId=="+subdetId+") && (layer=="+str(trackerLayer)+")"),
        )
        trackingRecHitProducerNoMerge.plugins.append(pluginConfig)

#for subdetId in ["BPX","FPX","TIB","TID","TOB","TEC"]:
#    plugin1Config = cms.PSet(
#        name = cms.string("monitor"+subdetId),
#        type=cms.string("TrackingRecHitMonitorPlugin"),
#        dxmax=cms.double(0.05),
#        dymax=cms.double(20.0),
#        select=cms.string("subdetId=="+subdetId),
#    )
#    trackingRecHitProducerNoMerge.plugins.append(plugin1Config)

#TFileService = cms.Service("TFileService", fileName = cms.string("histo.root") )
