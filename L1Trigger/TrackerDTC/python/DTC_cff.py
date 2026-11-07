# This describes the DTC Stub processing emulation
import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackerDTC.DTC_cfi import ProducerDTC_params
from L1Trigger.TrackerDTC.Setup_cff import TrackerDTCSetup

ProducerDTC = cms.EDProducer('trackerDTC::ProducerDTC', ProducerDTC_params)
