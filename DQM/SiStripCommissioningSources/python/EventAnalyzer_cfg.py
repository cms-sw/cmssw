import FWCore.ParameterSet.Config as cms

# process declaration
process = cms.Process("EventAnalyzer")

# message logger
process.load('DQM.SiStripCommissioningSources.OnlineMessageLogger_cff')

# DQM service
process.load('DQM.SiStripCommissioningSources.OnlineDQM_cff')

# input source
process.load('DQM.SiStripCommissioningSources.OnlineSource_cfi')

# event content analyzer
process.anal = cms.EDAnalyzer("EventContentAnalyzer")
process.p1 = cms.Path(process.anal)

# output
process.load('DQM.SiStripCommissioningSources.OnlineOutput_cfi')
process.outpath = cms.EndPath( process.consumer )
