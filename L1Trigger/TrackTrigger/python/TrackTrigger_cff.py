import FWCore.ParameterSet.Config as cms

#from Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi import *

from L1Trigger.TrackTrigger.TTClusterAlgorithmRegister_cfi import *
from L1Trigger.TrackTrigger.TTCluster_cfi import *

from L1Trigger.TrackTrigger.TTStubAlgorithmRegister_cfi import *
from L1Trigger.TrackTrigger.TTStub_cfi import *

from L1Trigger.TrackTrigger.TTTracks_cfi import *

TTClusterAlgorithm_Phase2TrackerDigi_ = cms.ESPrefer('TTClusterAlgorithm_2d2013_Phase2TrackerDigi_')
TTStubAlgorithm_Phase2TrackerDigi_ = cms.ESPrefer('TTStubAlgorithm_window2013_Phase2TrackerDigi_')

#and the sequence to run
TrackTriggerClustersStubs = cms.Sequence(TTClustersFromPhase2TrackerDigis*TTStubsFromPhase2TrackerDigis)

TrackTriggerTTTracks = cms.Sequence(BeamSpotFromSim*TTTracksFromPhase2TrackerDigis)
