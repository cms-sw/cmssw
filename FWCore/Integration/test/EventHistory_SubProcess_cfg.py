import FWCore.ParameterSet.Config as cms

process = cms.Process("FIRST")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("EmptySource")

process.intdeque = cms.EDProducer("IntDequeProducer",
    count = cms.int32(12),
    ivalue = cms.int32(21)
)

process.intlist = cms.EDProducer("IntListProducer",
    count = cms.int32(4),
    ivalue = cms.int32(3)
)

process.intset = cms.EDProducer("IntSetProducer",
    start = cms.int32(100),
    stop = cms.int32(110)
)

process.intvec = cms.EDProducer("IntVectorProducer",
    count = cms.int32(9),
    ivalue = cms.int32(11)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testEventHistory_1.root')
)

process.p1 = cms.Path(process.intdeque+process.intlist+process.intset+process.intvec)
process.ep1 = cms.EndPath(process.out)

process2 = cms.Process("SECOND")

process.subProcess = cms.SubProcess(process2)

process2.intdeque = cms.EDProducer("IntDequeProducer",
    count = cms.int32(12),
    ivalue = cms.int32(21)
)

process2.intlist = cms.EDProducer("IntListProducer",
    count = cms.int32(4),
    ivalue = cms.int32(3)
)

process2.intset = cms.EDProducer("IntSetProducer",
    start = cms.int32(100),
    stop = cms.int32(110)
)

process2.intvec = cms.EDProducer("IntVectorProducer",
    count = cms.int32(9),
    ivalue = cms.int32(11)
)

process2.filt55 = cms.EDFilter("TestFilterModule",
    acceptValue = cms.untracked.int32(55)
)

process2.filt75 = cms.EDFilter("TestFilterModule",
    acceptValue = cms.untracked.int32(75)
)

process2.out = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('f55')
    ),
    fileName = cms.untracked.string('testEventHistory_2.root')
)

process2.s = cms.Sequence(process2.intdeque+process2.intlist+process2.intset+process2.intvec)
process2.f55 = cms.Path(process2.s*process2.filt55)
process2.f75 = cms.Path(process2.s*process2.filt75)
process2.ep2 = cms.EndPath(process2.out)

process2.sched = cms.Schedule(process2.f55, process2.f75, process2.ep2)

process3 = cms.Process("THIRD")

process2.subProcess = cms.SubProcess(process3,
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('f55')
    )
)

process3.intdeque = cms.EDProducer("IntDequeProducer",
    count = cms.int32(12),
    ivalue = cms.int32(21)
)

process3.intlist = cms.EDProducer("IntListProducer",
    count = cms.int32(4),
    ivalue = cms.int32(3)
)

process3.intset = cms.EDProducer("IntSetProducer",
    start = cms.int32(100),
    stop = cms.int32(110)
)

process3.intvec = cms.EDProducer("IntVectorProducer",
    count = cms.int32(9),
    ivalue = cms.int32(11)
)

process3.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testEventHistory_3.root')
)

process3.outother = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testEventHistory_other.root')
)

process3.p3 = cms.Path(process3.intdeque+process3.intlist+process3.intset)
process3.ep31 = cms.EndPath(process3.out)
process3.ep32 = cms.EndPath(process3.intvec*process3.intset*process3.outother*process3.out*process3.outother)
process3.epother = cms.EndPath(process3.outother)

process4 = cms.Process("FOURTH")

process3.subProcess = cms.SubProcess(process4)

process4.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testEventHistory_4.root')
)

process4.ep4 = cms.EndPath(process4.out)

process5 = cms.Process("FIFTH")

process4.subProcess = cms.SubProcess(process5)

process5.out = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring('f55:SECOND','f75:SECOND')
    ),
    fileName = cms.untracked.string('testEventHistory_5.root')
)

process5.ep4 = cms.EndPath(process5.out)

process6 = cms.Process("SIXTH")

process5.subProcess = cms.SubProcess(process6,
    SelectEvents = cms.untracked.PSet(
       SelectEvents = cms.vstring('f55:SECOND','f75:SECOND')
    )
)

process6.historytest = cms.EDAnalyzer("HistoryAnalyzer",
    # Why does the filter module (from step 3) pass 56 events, when I
    # give it a rate of 55%? (after 100 events are created the filter
    # starts counting at 1 and passes events 1 to 55 then 100 also ...)
    expectedCount = cms.int32(56),
    # this does not count the current process
    expectedSize = cms.int32(4),
    # check SelectEventInfo from previous processes
    expectedSelectEventsInfo = cms.VPSet(
      cms.PSet(
        EndPathPositions = cms.vint32(),
        EndPaths =  cms.vstring(),
        InProcessHistory = cms.bool(True),
        SelectEvents = cms.vstring()
      ),
      cms.PSet(
        EndPathPositions = cms.vint32(),
        EndPaths =  cms.vstring(),
        InProcessHistory = cms.bool(True),
        SelectEvents = cms.vstring('f55')
      ),
      cms.PSet(
        EndPathPositions = cms.vint32(),
        EndPaths =  cms.vstring(),
        InProcessHistory = cms.bool(True),
        SelectEvents = cms.vstring()
      ),
      cms.PSet(
        EndPathPositions = cms.vint32(),
        EndPaths =  cms.vstring(),
        InProcessHistory = cms.bool(False),
        SelectEvents = cms.vstring('f55:SECOND','f75:SECOND')        
      )
    ),
    # check the deletion modules from the top level
    # ParameterSet of the current process. The following
    # should be deleted
    #    OutputModules
    #    EDAnalyzers not on endPaths
    #    if unscheduled, EDFilters and EDProducers not on paths or end paths
    # The module ParameterSet should be removed
    # The label should be removed from @all_modules and all end paths
    # empty end paths should be removed from @end_paths and @paths
    expectedPaths = cms.vstring('p1', 'p2', 'ep62'),
    expectedEndPaths = cms.vstring('ep62'),
    expectedModules = cms.vstring('analyzerOnPath', 'filterOnEndPath', 'filterOnPath', 'historytest', 'producerOnEndPath', 'producerOnPath'),
    expectedDroppedEndPaths = cms.vstring('ep61', 'ep63'),
    expectedDroppedModules = cms.vstring('dummyanalyzer', 'dummyfilter', 'dummyproducer', 'dummyout'),
    expectedDropFromProcPSet = cms.vstring('out', 'out2', 'analyzerOnEndPath'),
    expectedModulesOnEndPaths = cms.PSet(
      ep62 = cms.vstring('producerOnEndPath', 'filterOnEndPath', 'historytest')
    )
)

process6.producerOnPath = cms.EDProducer("IntProducer",
    ivalue = cms.int32(2)
)

process6.producerOnEndPath = cms.EDProducer("IntProducer",
    ivalue = cms.int32(2)
)

process6.filterOnPath = cms.EDFilter("TestFilterModule",
    acceptValue = cms.untracked.int32(55)
)

process6.filterOnEndPath = cms.EDFilter("TestFilterModule",
    acceptValue = cms.untracked.int32(55)
)

process6.analyzerOnPath = cms.EDAnalyzer("NonAnalyzer")

process6.analyzerOnEndPath = cms.EDAnalyzer("NonAnalyzer")

process6.dummyanalyzer = cms.EDAnalyzer("NonAnalyzer")

process6.dummyproducer = cms.EDProducer("IntProducer",
    ivalue = cms.int32(2)
)

process6.dummyfilter = cms.EDFilter("TestFilterModule",
    acceptValue = cms.untracked.int32(55)
)

process6.dummyout = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('dummy_6.root')
)

process6.out = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
      SelectEvents = cms.vstring('f55:SECOND','f75:SECOND')
    ),
    fileName = cms.untracked.string('testEventHistory_6.root')
)

process6.out2 = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
      SelectEvents = cms.vstring('f55:SECOND','f75:SECOND')
    ),
    fileName = cms.untracked.string('testEventHistory_62.root')
)

process6.p1 = cms.Path(process6.historytest*process6.filterOnPath*process6.producerOnPath)
process6.p2 = cms.Path(process6.historytest*process6.analyzerOnPath)

process6.ep61 = cms.EndPath(process6.out)
process6.ep62 = cms.EndPath(process6.producerOnEndPath*process6.filterOnEndPath*process6.out*process6.historytest)
process6.ep63 = cms.EndPath(process6.analyzerOnEndPath*process6.out2*process6.out)
