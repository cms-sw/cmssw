import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackerTFP.Demonstrator_cfi import TrackTriggerDemonstrator_params
from L1Trigger.TrackerTFP.Producer_cfi import TrackerTFPProducer_params

TrackTriggerDemonstrator = cms.ESProducer("trackerTFP::ProducerDemonstrator", TrackTriggerDemonstrator_params)

TrackerTFPDemonstrator = cms.EDAnalyzer("trackerTFP::AnalyzerDemonstrator", TrackTriggerDemonstrator_params, TrackerTFPProducer_params)