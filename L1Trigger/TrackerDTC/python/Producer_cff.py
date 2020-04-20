import FWCore.ParameterSet.Config as cms

#---------------------------------------------------------------------------------------------------------
# This describes the DTC Stub processing
#---------------------------------------------------------------------------------------------------------

#=== Import default values for all parameters & define EDProducer.

from L1Trigger.TrackerDTC.Producer_Defaults_cfi import TrackerDTCProducer_params
from L1Trigger.TrackerDTC.Format_Hybrid_cfi import TrackerDTCFormat_params
from L1Trigger.TrackerDTC.Analyzer_Defaults_cfi import TrackerDTCAnalyzer_params

TrackerDTCProducer = cms.EDProducer('trackerDTC::Producer', TrackerDTCProducer_params, TrackerDTCFormat_params, TrackerDTCAnalyzer_params)