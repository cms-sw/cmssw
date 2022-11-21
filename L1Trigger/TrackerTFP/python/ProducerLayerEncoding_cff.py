import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackerTFP.ProducerLayerEncoding_cfi import TrackTriggerLayerEncoding_params

TrackTriggerLayerEncoding = cms.ESProducer("trackerTFP::ProducerLayerEncoding", TrackTriggerLayerEncoding_params)