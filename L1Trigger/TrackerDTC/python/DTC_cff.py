import FWCore.ParameterSet.Config as cms

#---------------------------------------------------------------------------------------------------------
# This describes the DTC Stub processing
#---------------------------------------------------------------------------------------------------------

#=== Import default values for all parameters & define EDProducer.

from L1Trigger.TrackerDTC.DTC_cfi import TrackerDTC_params
from L1Trigger.TrackTrigger.Setup_cff import TrackTriggerSetup
from L1Trigger.TrackerDTC.LayerEncoding_cff import TrackerDTCLayerEncoding
from L1Trigger.TrackerTFP.DataFormats_cff import TrackTriggerDataFormats

ProducerDTC = cms.EDProducer('trackerDTC::ProducerDTC', TrackerDTC_params)
