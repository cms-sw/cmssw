import FWCore.ParameterSet.Config as cms

# Get required input ESProducts
from L1Trigger.TrackFindingTracklet.DataFormats_cff import TrackFindingTrackletDataFormats
from L1Trigger.TrackFindingTracklet.Setup_cff import TrackFindingTrackletSetup

# HitPatternHelper configuration
from L1Trigger.TrackFindingTracklet.ProducerHPH_cfi import HitPatternHelper_params

HitPatternHelperSetup = cms.ESProducer("hph::ProducerHPH", HitPatternHelper_params)
