import FWCore.ParameterSet.Config as cms
import DQMServices.Components.test.checkBooking as booking
import DQMServices.Components.test.createElements as c
import sys

process = cms.Process("TEST")

# load DQM
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")


b = booking.BookingParams(sys.argv)
b.doCheck(testOnly=False)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))
process.source = cms.Source("EmptySource", numberEventsInRun = cms.untracked.uint32(10),
                            firstLuminosityBlock = cms.untracked.uint32(1),
                            firstEvent = cms.untracked.uint32(1),
                            numberEventsInLuminosityBlock = cms.untracked.uint32(1))


elements = c.createElements()
readRunElements = c.createReadRunElements()
readLumiElements = c.createReadLumiElements()

process.filler = cms.EDAnalyzer("DummyBookFillDQMStore" + b.mt_postfix(),
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
process.dqmsave_step = cms.Path(process.dqmSaver)
process.o = cms.EndPath(process.out)

process.schedule = cms.Schedule(
    process.p,
    process.dqmsave_step
)

process.add_(cms.Service("DQMStore"))

if b.multithread():
    process.out.enableMultiThread = cms.untracked.bool(True)
    process.DQMStore.enableMultiThread = cms.untracked.bool(True)



if len(sys.argv) > 3:
    if sys.argv[3] == "ForceReset":
        print "Forcing Reset of histograms at every Run Transition."
        process.DQMStore.forceResetOnBeginRun = cms.untracked.bool(True)


#global options
process.DQMStore.verbose = cms.untracked.int32(5)

#process.dqmSaver.workflow = "A/B/C"
process.dqmSaver.convention = 'FilterUnit'
process.dqmSaver.saveByLumiSection = True
process.dqmSaver.fileFormat = cms.untracked.string('PB')

#process.add_(cms.Service("Tracer"))

