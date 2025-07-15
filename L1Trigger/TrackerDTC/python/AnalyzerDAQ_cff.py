# EDAnalyzer to analyze TTCluster Occupancies on DTCs, plots cluster occupancy

import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackerDTC.AnalyzerDAQ_cfi import TrackerDTCAnalyzerDAQ_params
from L1Trigger.TrackTrigger.Setup_cff import TrackTriggerSetup

TrackerDTCAnalyzerDAQ = cms.EDAnalyzer('trackerDTC::AnalyzerDAQ', TrackerDTCAnalyzerDAQ_params)
