import FWCore.ParameterSet.Config as cms

from Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi import *

from L1Trigger.TrackTrigger.TTClusterAlgorithmRegister_cfi import *
from L1Trigger.TrackTrigger.TTCluster_cfi import *

from L1Trigger.TrackTrigger.TTStubAlgorithmRegister_cfi import *
from L1Trigger.TrackTrigger.TTStub_cfi import *

from L1Trigger.TrackTrigger.TTTrackAlgorithmRegister_cfi import *
from L1Trigger.TrackTrigger.TTTrack_cfi import *

TTClusterAlgorithm_PixelDigi_ = cms.ESPrefer('TTClusterAlgorithm_2d2013_PixelDigi_')
TTStubAlgorithm_PixelDigi_ = cms.ESPrefer('TTStubAlgorithm_tab2013_PixelDigi_')
TTTrackAlgorithm_PixelDigi_ = cms.ESPrefer('TTTrackAlgorithm_trackletBE_PixelDigi_')

#and the sequence to run
TrackTriggerClustersStubs = cms.Sequence(TTClustersFromPixelDigis*TTStubsFromPixelDigis)
TrackTriggerTracks = cms.Sequence(TTTracksFromPixelDigis)
TrackTriggerComplete = cms.Sequence(TTClustersFromPixelDigis*TTStubsFromPixelDigis*TTTracksFromPixelDigis)

