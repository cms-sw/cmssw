# The following comments couldn't be translated into the new config version:

# SEEDS

import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFTracking.nuclear_cff import *
import copy
from RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi import *
#TRACKER HITS
nuclearPixelRecHits = copy.deepcopy(siPixelRecHits)
import copy
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi import *
nuclearStripRecHits = copy.deepcopy(siStripMatchedRecHits)
import copy
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
#TRAJECTORY MEASUREMENT
nuclearMeasurementTracker = copy.deepcopy(MeasurementTracker)
#HIT REMOVAL
nuclearClusters = cms.EDProducer("RemainingClusterProducer",
    matchedRecHits = cms.InputTag("fourthStripRecHits","matchedRecHit"),
    recTracks = cms.InputTag("fourthvtxFilt"),
    stereorecHits = cms.InputTag("fourthStripRecHits","stereoRecHit"),
    rphirecHits = cms.InputTag("fourthStripRecHits","rphiRecHit"),
    pixelHits = cms.InputTag("fourthPixelRecHits")
)

nuclearRemainingHits = cms.Sequence(nuclearClusters*nuclearPixelRecHits*nuclearStripRecHits*nuclear)
nuclearPixelRecHits.src = 'nuclearClusters'
nuclearStripRecHits.ClusterProducer = 'nuclearClusters'
nuclearMeasurementTracker.ComponentName = 'nuclearMeasurementTracker'
nuclearMeasurementTracker.pixelClusterProducer = 'nuclearClusters'
nuclearMeasurementTracker.stripClusterProducer = 'nuclearClusters'
firstnuclearSeed.MeasurementTrackerName = 'nuclearMeasurementTracker'
secondnuclearSeed.MeasurementTrackerName = 'nuclearMeasurementTracker'
thirdnuclearSeed.MeasurementTrackerName = 'nuclearMeasurementTracker'
fourthnuclearSeed.MeasurementTrackerName = 'nuclearMeasurementTracker'
#TRAJECTORY BUILDER
nuclearCkfTrajectoryBuilder.MeasurementTrackerName = 'nuclearMeasurementTracker'

