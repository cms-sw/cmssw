import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackerDTC.Analyzer_Defaults_cfi import TrackerDTCAnalyzer_params
from L1Trigger.TrackerDTC.Producer_Defaults_cfi import TrackerDTCProducer_params
from L1Trigger.TrackerDTC.Format_Hybrid_cfi import TrackerDTCFormat_params

TrackerDTCAnalyzer = cms.EDAnalyzer('trackerDTC::Analyzer', TrackerDTCAnalyzer_params, TrackerDTCProducer_params, TrackerDTCFormat_params)