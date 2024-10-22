import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:overlap.root"))
process.tst = cms.EDAnalyzer("RunLumiEventChecker",
                             eventSequence = cms.untracked.VEventID(
        cms.EventID(1,0,0),
        cms.EventID(1,1,0),
        cms.EventID(1,1,1),
        cms.EventID(1,1,2),
        cms.EventID(1,1,3),
        cms.EventID(1,1,0),
        cms.EventID(1,2,0),
        cms.EventID(1,2,4),
        cms.EventID(1,2,5),
        cms.EventID(1,2,6),
        cms.EventID(1,2,0),
        cms.EventID(1,3,0),
        cms.EventID(1,3,7),
        cms.EventID(1,3,8),
        cms.EventID(1,3,9),
        cms.EventID(1,3,0),
        cms.EventID(1,4,0),
        cms.EventID(1,4,10),
        cms.EventID(1,4,0),
        cms.EventID(1,0,0)
        ),
                             unorderedEvents = cms.untracked.bool(True))

process.out = cms.EndPath(process.tst)
