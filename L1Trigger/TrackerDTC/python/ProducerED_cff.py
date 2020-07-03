import FWCore.ParameterSet.Config as cms

#---------------------------------------------------------------------------------------------------------
# This describes the DTC Stub processing
#---------------------------------------------------------------------------------------------------------

#=== Import default values for all parameters & define EDProducer.

from L1Trigger.TrackerDTC.ProducerED_cfi import TrackerDTCProducer_params
from L1Trigger.TrackerDTC.ProducerES_cff import TrackTriggerSetup

TrackerDTCProducer = cms.EDProducer('trackerDTC::ProducerED', TrackerDTCProducer_params)