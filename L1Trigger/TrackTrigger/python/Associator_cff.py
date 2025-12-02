import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackTrigger.Setup_cff import TrackTriggerSetup
from L1Trigger.TrackTrigger.Associator_cfi import Associator_params

Associator = cms.ESProducer('tt::ProducerAssociator', Associator_params)
