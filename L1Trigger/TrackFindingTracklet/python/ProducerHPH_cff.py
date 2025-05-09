import FWCore.ParameterSet.Config as cms

# Get required input ESProducts
from L1Trigger.TrackerTFP.DataFormats_cff import TrackTriggerDataFormats
from L1Trigger.TrackerTFP.LayerEncoding_cff import TrackTriggerLayerEncoding
from L1Trigger.TrackTrigger.Setup_cff import TrackTriggerSetup

# HitPatternHelper configuration
from L1Trigger.TrackFindingTracklet.ProducerHPH_cfi import HitPatternHelper_params

HitPatternHelperSetup = cms.ESProducer("hph::ProducerHPH", HitPatternHelper_params)
