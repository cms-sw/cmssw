import FWCore.ParameterSet.Config as cms

from FastSimulation.TrackingRecHitProducer.PixelPluginsPhase0_cfi import pixelPluginsPhase0
from FastSimulation.TrackingRecHitProducer.PixelPluginsPhase1_cfi import pixelPluginsPhase1
from FastSimulation.TrackingRecHitProducer.PixelPluginsPhase2_cfi import pixelPluginsPhase2

fastTrackerRecHits = cms.EDProducer("TrackingRecHitProducer",
    simHits = cms.InputTag("fastSimProducer","TrackerHits"),
    plugins = pixelPluginsPhase0
)

from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(fastTrackerRecHits, plugins = pixelPluginsPhase1)

from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(fastTrackerRecHits, plugins = pixelPluginsPhase2)

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
