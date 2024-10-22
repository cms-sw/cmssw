import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackFindingTracklet.Demonstrator_cfi import TrackTriggerDemonstrator_params
from L1Trigger.TrackFindingTracklet.Producer_cfi import TrackFindingTrackletProducer_params

TrackTriggerDemonstrator = cms.ESProducer("trackerTFP::ProducerDemonstrator", TrackTriggerDemonstrator_params)

TrackerTFPDemonstrator = cms.EDAnalyzer("trklet::AnalyzerDemonstrator", TrackTriggerDemonstrator_params, TrackFindingTrackletProducer_params)
