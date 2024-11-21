# ESProducer processing and providing run-time constants used by Track Trigger emulators

import FWCore.ParameterSet.Config as cms
from L1Trigger.TrackTrigger.Setup_cfi import TrackTrigger_params

TrackTriggerSetup = cms.ESProducer("tt::ProducerSetup", TrackTrigger_params)
