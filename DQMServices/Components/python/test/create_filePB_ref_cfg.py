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
process.source = cms.Source("EmptySource",
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
                               fileName = cms.untracked.string("dqm_filePB_ref.root"))


process.p = cms.Path(process.filler)
process.o = cms.EndPath(process.out)

process.schedule = cms.Schedule(
    process.p,
    process.o
)

process.add_(cms.Service("DQMStore"))

if b.multithread():
    process.out.enableMultiThread = cms.untracked.bool(True)
    process.DQMStore.enableMultiThread = cms.untracked.bool(True)
    process.options = cms.untracked.PSet(
        numberOfThreads = cms.untracked.uint32(4),
        numberOfStreams = cms.untracked.uint32(4)
        )




if len(sys.argv) > 3:
    if sys.argv[3] == "ForceReset":
        print "Forcing Reset of histograms at every Run Transition."
        process.DQMStore.forceResetOnBeginRun = cms.untracked.bool(True)


#----------------------------------------------------------#                                                                                                                                      
### global options Online ###                                                                                                                                                                       
process.DQMStore.LSbasedMode = cms.untracked.bool(False)
process.DQMStore.verbose = cms.untracked.int32(5)

process.dqmSaver.workflow = ''
process.dqmSaver.convention = 'FilterUnit'
process.dqmSaver.saveByLumiSection = -1
process.dqmSaver.fileFormat = cms.untracked.string('ROOT')
process.dqmSaver.fakeFilterUnitMode = cms.untracked.bool(True)

#process.add_(cms.Service("Tracer"))              
