import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackTrigger.ProducerSetup_cfi import TrackTrigger_params

TrackTriggerSetup = cms.ESProducer("tt::ProducerSetup", TrackTrigger_params)