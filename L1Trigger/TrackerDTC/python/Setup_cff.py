# This provides configuration used by DTC emulator
import FWCore.ParameterSet.Config as cms
from L1Trigger.TrackerDTC.Setup_cfi import TrackerDTCSetup_params

TrackerDTCSetup = cms.ESProducer("trackerDTC::ProducerSetup", TrackerDTCSetup_params)
