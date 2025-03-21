import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackerDTC.Analyzer_cfi import TrackerDTCAnalyzer_params
from L1Trigger.TrackerDTC.ProducerED_cfi import TrackerDTCProducer_params
from L1Trigger.TrackTrigger.ProducerSetup_cff import TrackTriggerSetup

TrackerDTCAnalyzer = cms.EDAnalyzer('trackerDTC::Analyzer', TrackerDTCAnalyzer_params, TrackerDTCProducer_params)
