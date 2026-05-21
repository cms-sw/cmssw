import FWCore.ParameterSet.Config as cms

ppsanalyzer = cms.EDAnalyzer(
   "PPSAnalyzerHC",
   ctppsLocalTracks    = cms.InputTag('ctppsLocalTrackLiteProducer')
)
