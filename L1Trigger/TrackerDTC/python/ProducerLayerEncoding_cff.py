import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackerDTC.ProducerLayerEncoding_cfi import TrackerDTCLayerEncoding_params

TrackerDTCLayerEncoding = cms.ESProducer("trackerDTC::ProducerLayerEncoding", TrackerDTCLayerEncoding_params)