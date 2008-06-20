import FWCore.ParameterSet.Config as cms

from DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff import *
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *
# tracker
from RecoLocalTracker.Configuration.RecoLocalTracker_Cosmics_cff import *
from RecoTracker.Configuration.RecoTrackerP5_cff import *
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
    connect = cms.string('frontier://Frontier/CMS_COND_20X_BEAMSPOT')
)

# from "Configuration/GlobalRuns/data/ReconstructionGR.cff" & "Configuration/StandardSequences/data/RawToDigi.cff"
trackerGR = cms.Sequence(siStripDigis*offlineBeamSpot*striptrackerlocalreco*ctftracksP5)
DQMSiStripMonitorTrack_Real = cms.Sequence(trackerGR*SiStripMonitorTrack)
siStripDigis.ProductLabel = 'source'

