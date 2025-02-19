import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.source = cms.Source("EmptySource"
                            )

process.demo = cms.EDAnalyzer('RPCDetIdAnalyzer'
)


process.p = cms.Path(process.demo)
