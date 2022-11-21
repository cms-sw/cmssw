import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackerDTC.AnalyzerDAQ_cfi import TrackerDTCAnalyzerDAQ_params

TrackerDTCAnalyzerDAQ = cms.EDAnalyzer('trackerDTC::AnalyzerDAQ', TrackerDTCAnalyzerDAQ_params)