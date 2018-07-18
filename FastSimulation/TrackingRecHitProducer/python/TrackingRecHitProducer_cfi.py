import FWCore.ParameterSet.Config as cms

import six

fastTrackerRecHits = cms.EDProducer("TrackingRecHitProducer",
    simHits = cms.InputTag("fastSimProducer","TrackerHits"),
    plugins=cms.VPSet()
)

fastTrackerRecHits.plugins.append(
    cms.PSet(
        name = cms.string("pixelBarrelSmearer"),
        type=cms.string("PixelBarrelTemplateSmearerPlugin"),
        BigPixelBarrelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/NewPixelResolutionBarrel38T.root'),
        EdgePixelBarrelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/NewPixelResolutionBarrelEdge38T.root'),
        RegularPixelBarrelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelBarrelResolution2014.root'),
        BigPixelForwardResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/NewPixelResolutionForward38T.root'),
        EdgePixelForwardResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/NewPixelResolutionForward38T.root'),
        RegularPixelForwardResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelForwardResolution2014.root'),
        UseCMSSWPixelParametrization = cms.bool(False),
        MergeHitsOn = cms.bool(False),
        MergingProbabilityBarrelFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bmergeprob.root'),
        MergingProbabilityForwardFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fmergeprob.root'),
        MergedPixelBarrelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bxsmear.root'),
        MergedPixelForwardResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fxsmear.root'),
        MergedPixelBarrelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bysmear.root'),
        MergedPixelForwardResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fysmear.root'),
        templateId = cms.int32( 40 ),
        select=cms.string("subdetId==BPX"),
    )
)

fastTrackerRecHits.plugins.append(
    cms.PSet(
        name = cms.string("pixelForwardSmearer"),
        type=cms.string("PixelForwardTemplateSmearerPlugin"),
        BigPixelBarrelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/NewPixelResolutionBarrel38T.root'),
        EdgePixelBarrelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/NewPixelResolutionBarrelEdge38T.root'),
        RegularPixelBarrelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelBarrelResolution2014.root'),
        BigPixelForwardResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/NewPixelResolutionForward38T.root'),
        EdgePixelForwardResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/NewPixelResolutionForward38T.root'),
        RegularPixelForwardResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelForwardResolution2014.root'),
        UseCMSSWPixelParametrization = cms.bool(False),
        MergeHitsOn = cms.bool(False),
        MergingProbabilityBarrelFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bmergeprob.root'),
        MergingProbabilityForwardFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fmergeprob.root'),
        MergedPixelBarrelResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bxsmear.root'),
        MergedPixelForwardResolutionXFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fxsmear.root'),
        MergedPixelBarrelResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/bysmear.root'),
        MergedPixelForwardResolutionYFile = cms.string('FastSimulation/TrackingRecHitProducer/data/fysmear.root'),
        templateId = cms.int32( 41 ),
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
        8: cms.double(0.00434),
        9: cms.double(0.00432),
    }
}

for subdetId,trackerLayers in six.iteritems(trackerStripGaussianResolutions):
    for trackerLayer, resolutionX in six.iteritems(trackerLayers):
        pluginConfig = cms.PSet(
            name = cms.string(subdetId+str(trackerLayer)),
            type=cms.string("TrackingRecHitStripGSPlugin"),
            resolutionX=resolutionX,
            select=cms.string("(subdetId=="+subdetId+") && (layer=="+str(trackerLayer)+")"),
        )
        fastTrackerRecHits.plugins.append(pluginConfig)
