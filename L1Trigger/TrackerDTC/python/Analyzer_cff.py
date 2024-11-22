# EDAnalyzer for hardware like structured TTStub Collection used by Track Trigger emulators, runs DTC stub emulation, plots performance & stub occupancy

import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackerDTC.Analyzer_cfi import TrackerDTCAnalyzer_params
from L1Trigger.TrackerDTC.DTC_cfi import TrackerDTC_params
from L1Trigger.TrackTrigger.Setup_cff import TrackTriggerSetup

AnalyzerDTC = cms.EDAnalyzer('trackerDTC::Analyzer', TrackerDTCAnalyzer_params, TrackerDTC_params)
