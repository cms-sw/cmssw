import FWCore.ParameterSet.Config as cms

process = cms.Process("SIXTH")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:testEventHistory_5.root')
)

process.historytest = cms.EDAnalyzer("HistoryAnalyzer",
    # Why does the filter module (from step 3) pass 56 events, when I
    # give it a rate of 55%? (after 100 events are created the filter
    # starts counting at 1 and passes events 1 to 55 then 100 also ...)
    expectedCount = cms.int32(56),
    # this does not count the current process
    expectedSize = cms.int32(4),
    # check SelectEventInfo from previous processes
    expectedSelectEventsInfo = cms.VPSet(
      cms.PSet(
        EndPathPositions = cms.vint32(0),
        EndPaths =  cms.vstring('ep1'),
        InProcessHistory = cms.bool(True),
        SelectEvents = cms.vstring()
      ),
      cms.PSet(
        EndPathPositions = cms.vint32(0),
        EndPaths =  cms.vstring('ep2'),
        InProcessHistory = cms.bool(True),
        SelectEvents = cms.vstring('f55')
      ),
      cms.PSet(
        EndPathPositions = cms.vint32(0, 2),
        EndPaths =  cms.vstring('ep31', 'ep32'),
        InProcessHistory = cms.bool(True),
        SelectEvents = cms.vstring()
      ),
      cms.PSet(
        EndPathPositions = cms.vint32(0),
        EndPaths =  cms.vstring('ep4'),
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

process.producerOnPath = cms.EDProducer("IntProducer",
    ivalue = cms.int32(2)
)

process.producerOnEndPath = cms.EDProducer("IntProducer",
    ivalue = cms.int32(2)
)

process.filterOnPath = cms.EDFilter("TestFilterModule",
    acceptValue = cms.untracked.int32(55)
)

process.filterOnEndPath = cms.EDFilter("TestFilterModule",
    acceptValue = cms.untracked.int32(55)
)

process.analyzerOnPath = cms.EDAnalyzer("NonAnalyzer")

process.analyzerOnEndPath = cms.EDAnalyzer("NonAnalyzer")

process.dummyanalyzer = cms.EDAnalyzer("NonAnalyzer")

process.dummyproducer = cms.EDProducer("IntProducer",
    ivalue = cms.int32(2)
)

process.dummyfilter = cms.EDFilter("TestFilterModule",
    acceptValue = cms.untracked.int32(55)
)

process.dummyout = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('dummy_6.root')
)

process.out = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
      SelectEvents = cms.vstring('f55:SECOND','f75:SECOND')
    ),
    fileName = cms.untracked.string('testEventHistory_6.root')
)

process.out2 = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
      SelectEvents = cms.vstring('f55:SECOND','f75:SECOND')
    ),
    fileName = cms.untracked.string('testEventHistory_62.root')
)

process.p1 = cms.Path(process.historytest*process.filterOnPath*process.producerOnPath)
process.p2 = cms.Path(process.historytest*process.analyzerOnPath)

process.ep61 = cms.EndPath(process.out)
process.ep62 = cms.EndPath(process.producerOnEndPath*process.filterOnEndPath*process.out*process.historytest)
process.ep63 = cms.EndPath(process.analyzerOnEndPath*process.out2*process.out)
