import FWCore.ParameterSet.Config as cms

PostProcessorHcalIsoTrack = cms.EDAnalyzer("DQMHcalIsoTrackPostProcessor",
     subDir = cms.untracked.string("AlCaReco/HcalIsoTrack"),
     saveToFile=cms.bool(False),
     outputFile=cms.string("HarvestOutput.root")
)
