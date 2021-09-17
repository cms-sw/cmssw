import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.source = cms.Source("DQMRootSource",
                            reScope = cms.untracked.string(""),
                            fileNames = cms.untracked.vstring("file:dqm_missing.root"))

seq = cms.untracked.VEventID()

process.check = cms.EDAnalyzer("RunLumiEventChecker",
                               eventSequence = seq)

readRunElements = list()


readLumiElements=list()

process.reader = cms.EDAnalyzer("DummyReadDQMStore",
                                 runElements = cms.untracked.VPSet(*readRunElements),
                                 lumiElements = cms.untracked.VPSet(*readLumiElements) )

process.e = cms.EndPath(process.check+process.reader)

process.add_(cms.Service("DQMStore"))
#process.add_(cms.Service("Tracer"))

