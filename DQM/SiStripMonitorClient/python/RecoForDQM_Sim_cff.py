import FWCore.ParameterSet.Config as cms

# Digitiser ####
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *
# Zero Suppression  ####
from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi import *
# Cluster Finder ####
from RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
# TrackRefitter ####
from RecoTracker.TrackProducer.RefitterWithMaterial_cff import *
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
BeamSpotEarlyCollision = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0)
    ),
    timetype = cms.string('runtime'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('BeamSpotObjectsRcd'),
        tag = cms.string('EarlyCollision_5p3cm_mc')
    )),
    connect = cms.string('frontier://Frontier/CMS_COND_20X_BEAMSPOT') ##Frontier/CMS_COND_20X_BEAMSPOT"

)

RecoModulesForSimData = cms.Sequence(siStripDigis*siStripZeroSuppression*siStripClusters*offlineBeamSpot*TrackRefitter)
TrackRefitter.TrajectoryInEvent = True

