import FWCore.ParameterSet.Config as cms

#---------------------------------------------------------------------------------------------------------
# This describes the DTC Stub processing
#---------------------------------------------------------------------------------------------------------

#=== Import default values for all parameters & define EDProducer.

from L1Trigger.TrackerDTC.Producer_Defaults_cfi import TrackerDTCProducer_params

TrackerDTCProducer = cms.EDProducer( 'TrackerDTCProducer', TrackerDTCProducer_params )

if TrackerDTCProducer_params.ParamsED.DataFormat == "Hybrid":
  from L1Trigger.TrackerDTC.Format_Hybrid_cfi import TrackerDTCFormat_params
  TrackerDTCProducer = cms.EDProducer('TrackerDTCProducer', TrackerDTCProducer_params, TrackerDTCFormat_params )
elif TrackerDTCProducer_params.ParamsED.DataFormat == "TMTT":
  from L1Trigger.TrackerDTC.Format_TMTT_cfi import TrackerDTCFormat_params
  TrackerDTCProducer = cms.EDProducer('TrackerDTCProducer', TrackerDTCProducer_params, TrackerDTCFormat_params )