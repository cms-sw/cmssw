# The following comments couldn't be translated into the new config version:

# SEEDS

import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFTracking.nuclear_cff import *
import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
#TRACKER HITS
nuclearPixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone()
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
nuclearStripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone()
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
#TRAJECTORY MEASUREMENT
nuclearMeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone()
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

