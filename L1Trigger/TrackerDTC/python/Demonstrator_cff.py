# This compares DTC emulation with f/w

import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackerDTC.Demonstrator_cfi import TrackerDTCDemonstrator_params
from L1Trigger.TrackerDTC.Setup_cff import TrackerDTCSetup

TrackerDTCDemonstrator = cms.EDAnalyzer('trackerDTC::Demonstrator', TrackerDTCDemonstrator_params )
