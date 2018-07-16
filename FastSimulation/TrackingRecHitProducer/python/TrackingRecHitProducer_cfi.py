# Python 2 vs 3 compatibility library:
import six

import FWCore.ParameterSet.Config as cms


# Load the detailed configurations of pixel plugins.
# NB: for any new detector geometry (e.g. Phase 2 varians), we should write a new plugin
# config file, and import it here, and below use its own Era to load it.
#
from FastSimulation.TrackingRecHitProducer.PixelPluginsPhase0_cfi import pixelPluginsPhase0
from FastSimulation.TrackingRecHitProducer.PixelPluginsPhase1_cfi import pixelPluginsPhase1
from FastSimulation.TrackingRecHitProducer.PixelPluginsPhase2_cfi import pixelPluginsPhase2

# The default is (for better of worse) Phase 0:
#
fastTrackerRecHits = cms.EDProducer("TrackingRecHitProducer",
    simHits = cms.InputTag("fastSimProducer","TrackerHits"),
    plugins = pixelPluginsPhase0
)

# Phase 1 Era: replace plugins by Phase 1 plugins
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(fastTrackerRecHits, plugins = pixelPluginsPhase1)

# Phase 2 Era: replace plugins by Phase 2 plugins, etc...
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(fastTrackerRecHits, plugins = pixelPluginsPhase2)

# Configure strip tracker Gaussian-smearing plugins:
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
