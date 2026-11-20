import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackerDTC.Setup_cff import TrackerDTCSetup
from L1Trigger.TrackTrigger.Associator_cfi import Associator_params

Associator = cms.ESProducer('tt::ProducerAssociator', Associator_params)
