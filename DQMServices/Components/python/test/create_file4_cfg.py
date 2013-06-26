import FWCore.ParameterSet.Config as cms
import DQMServices.Components.test.checkBooking as booking
import DQMServices.Components.test.createElements as c
import sys

process = cms.Process("TEST")

b = booking.BookingParams(sys.argv)
b.doCheck(testOnly=False)

process.source = cms.Source("EmptySource", numberEventsInRun = cms.untracked.uint32(10),
                            firstLuminosityBlock = cms.untracked.uint32(1),
                            firstEvent = cms.untracked.uint32(1),
                            numberEventsInLuminosityBlock = cms.untracked.uint32(1))

elements = c.createElements()
readRunElements = c.createReadRunElements()
readLumiElements = c.createReadLumiElements()

process.filler = cms.EDAnalyzer("DummyBookFillDQMStore",
                                folder    = cms.untracked.string("TestFolder/"),
                                elements  = cms.untracked.VPSet(*elements),
                                fillRuns  = cms.untracked.bool(True),
                                fillLumis = cms.untracked.bool(True),
                                book_at_constructor = cms.untracked.bool(b.getBookLogic('CTOR')),
                                book_at_beginJob = cms.untracked.bool(b.getBookLogic('BJ')),
                                book_at_beginRun = cms.untracked.bool(b.getBookLogic('BR')))

process.out = cms.OutputModule("DQMRootOutputModule",
                               fileName = cms.untracked.string("dqm_file4.root"))


process.p = cms.Path(process.filler)
process.o = cms.EndPath(process.out)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(20))

process.add_(cms.Service("DQMStore"))

if len(sys.argv) > 3:
    if sys.argv[3] == "ForceReset": 
        print "Forcing Reset of histograms at every Run Transition."
        process.DQMStore.forceResetOnBeginRun = cms.untracked.bool(True)

#process.DQMStore.verbose = cms.untracked.int32(2)
#process.add_(cms.Service("Tracer"))

