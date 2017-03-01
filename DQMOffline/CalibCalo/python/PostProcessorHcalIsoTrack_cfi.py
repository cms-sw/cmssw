import FWCore.ParameterSet.Config as cms

PostProcessorHcalIsoTrack = cms.EDAnalyzer("DQMHcalIsoTrackPostProcessor",
     subDir = cms.untracked.string("AlCaReco/HcalIsoTrack"),
)
