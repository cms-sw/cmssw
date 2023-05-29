import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackTrigger.ProducerHPH_cfi import HitPatternHelper_params

HitPatternHelperSetup = cms.ESProducer("hph::ProducerHPH", HitPatternHelper_params)
