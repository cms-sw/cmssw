import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.source = cms.Source("OneLumiPoolSource", fileNames = cms.untracked.vstring("file:multi_lumi.root") )

process.tst = cms.EDAnalyzer("RunLumiEventChecker",
                             eventSequence = cms.untracked.VEventID(
                                                                    cms.EventID(1,0,0),
                                                                    cms.EventID(1,1,0),
                                                                    cms.EventID(1,1,1),
                                                                    cms.EventID(1,1,2),
                                                                    cms.EventID(1,1,3),
                                                                    cms.EventID(1,1,4),
                                                                    cms.EventID(1,1,5),
                                                                    cms.EventID(1,1,6),
                                                                    cms.EventID(1,1,7),
                                                                    cms.EventID(1,1,8),
                                                                    cms.EventID(1,1,9),
                                                                    cms.EventID(1,1,10),
                                                                    cms.EventID(1,1,11),
                                                                    cms.EventID(1,1,12),
                                                                    cms.EventID(1,1,13),
                                                                    cms.EventID(1,1,14),
                                                                    cms.EventID(1,1,15),
                                                                    cms.EventID(1,1,16),
                                                                    cms.EventID(1,1,17),
                                                                    cms.EventID(1,1,18),
                                                                    cms.EventID(1,1,19),
                                                                    cms.EventID(1,1,20),
                                                                    cms.EventID(1,1,0),
                                                                    cms.EventID(1,0,0)
                             )
                             )
process.dump = cms.OutputModule("AsciiOutputModule")

process.d = cms.EndPath(process.dump+process.tst)

#process.add_(cms.Service("Tracer"))
