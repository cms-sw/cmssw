import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackerDTC.ProducerES_cfi import TrackTrigger_params

TrackTriggerSetup = cms.ESProducer("trackerDTC::ProducerES", TrackTrigger_params)