import FWCore.ParameterSet.Config as cms

#clusters
from L1Trigger.TrackTrigger.TTClusterAlgorithmRegister_cfi import *
from L1Trigger.TrackTrigger.TTCluster_cfi import *

#stubs
from L1Trigger.TrackTrigger.TTStubAlgorithmRegister_cfi import *
from L1Trigger.TrackTrigger.TTStub_cfi import *

#prefered algos
TTClusterAlgorithm_Phase2TrackerDigi_ = cms.ESPrefer('TTClusterAlgorithm_official_Phase2TrackerDigi_')
TTStubAlgorithm_Phase2TrackerDigi_ = cms.ESPrefer('TTStubAlgorithm_official_Phase2TrackerDigi_')

#sequence
TrackTriggerClustersStubs = cms.Sequence(TTClustersFromPhase2TrackerDigis*TTStubsFromPhase2TrackerDigis)

