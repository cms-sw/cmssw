import FWCore.ParameterSet.Config as cms

PostProcessorHcalIsoTrack = cms.EDProducer("DQMHcalIsoTrackPostProcessor",
     subDir = cms.untracked.string("AlCaReco/HcalIsoTrack"),
)
