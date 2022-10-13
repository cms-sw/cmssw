import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackerTFP.ProducerES_cfi import TrackTriggerDataFormats_params

TrackTriggerDataFormats = cms.ESProducer("trackerTFP::ProducerES", TrackTriggerDataFormats_params)