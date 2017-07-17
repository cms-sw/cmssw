import FWCore.ParameterSet.Config as cms

process =cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.tester = cms.EDAnalyzer("AbortOnEventIDAnalyzer",
                                eventsToAbort = cms.untracked.VEventID( cms.EventID(1,10) ),
                                throwExceptionInsteadOfAbort = cms.untracked.bool(True))

process.p = cms.Path(process.tester)

process.add_(cms.Service("Tracer"))
