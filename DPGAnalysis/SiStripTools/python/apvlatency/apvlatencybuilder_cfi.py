import FWCore.ParameterSet.Config as cms

apvLatencyBuilder = cms.EDAnalyzer('APVLatencyBuilder',
                                   latencyIOVs = cms.VPSet()
                                   )
