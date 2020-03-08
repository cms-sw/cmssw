import FWCore.ParameterSet.Config as cms

#---------------------------------------------------------------------------------------------------------
# This describes the DTC Stub processing
#---------------------------------------------------------------------------------------------------------

#=== Import default values for all parameters & define EDProducer.

from L1Trigger.L1TTrackerDTC.Producer_Defaults_cfi import L1TTrackerDTCProducer_params

L1TTrackerDTCProducer = cms.EDProducer( 'L1TTrackerDTCProducer', L1TTrackerDTCProducer_params )

if L1TTrackerDTCProducer_params.ParamsED.DataFormat == "Hybrid":
  from L1Trigger.L1TTrackerDTC.Format_Hybrid_cfi import L1TTrackerDTCFormat_params
  L1TTrackerDTCProducer = cms.EDProducer('L1TTrackerDTCProducer', L1TTrackerDTCProducer_params, L1TTrackerDTCFormat_params )
elif L1TTrackerDTCProducer_params.ParamsED.DataFormat == "TMTT":
  from L1Trigger.L1TTrackerDTC.Format_TMTT_cfi import L1TTrackerDTCFormat_params
  L1TTrackerDTCProducer = cms.EDProducer('L1TTrackerDTCProducer', L1TTrackerDTCProducer_params, L1TTrackerDTCFormat_params )