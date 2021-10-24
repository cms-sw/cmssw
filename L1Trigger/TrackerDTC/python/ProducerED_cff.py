import FWCore.ParameterSet.Config as cms

#---------------------------------------------------------------------------------------------------------
# This describes the DTC Stub processing
#---------------------------------------------------------------------------------------------------------

#=== Import default values for all parameters & define EDProducer.

from L1Trigger.TrackerDTC.ProducerED_cfi import TrackerDTCProducer_params
from L1Trigger.TrackTrigger.ProducerSetup_cff import TrackTriggerSetup
from L1Trigger.TrackerDTC.ProducerLayerEncoding_cff import TrackerDTCLayerEncoding

TrackerDTCProducer = cms.EDProducer('trackerDTC::ProducerED', TrackerDTCProducer_params)