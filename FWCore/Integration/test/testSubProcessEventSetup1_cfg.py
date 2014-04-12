import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.INFO.limit = 200

#process.options = cms.untracked.PSet(forceEventSetupCacheClearOnNewRun = cms.untracked.bool(True))

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(1),
    firstLuminosityBlock = cms.untracked.uint32(1),
    firstEvent = cms.untracked.uint32(1),
    numberEventsInLuminosityBlock = cms.untracked.uint32(1),
    numberEventsInRun = cms.untracked.uint32(1)
)

process.emptyESSourceB = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordB"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process.emptyESSourceC = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordC"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process.emptyESSourceD = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordD"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process.emptyESSourceE = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordE"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process.emptyESSourceF = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordF"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process.emptyESSourceG = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordG"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process.emptyESSourceH = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordH"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process.emptyESSourceI = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordI"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process.emptyESSourceJ = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordJ"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process.emptyESSourceK = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordK"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process.esTestProducerB = cms.ESProducer("ESTestProducerB")
process.esTestProducerC = cms.ESProducer("ESTestProducerC")
process.esTestProducerD = cms.ESProducer("ESTestProducerD")
process.esTestProducerE = cms.ESProducer("ESTestProducerE")
process.esTestProducerF = cms.ESProducer("ESTestProducerF")
process.esTestProducerG = cms.ESProducer("ESTestProducerG")
process.esTestProducerH = cms.ESProducer("ESTestProducerH")
process.esTestProducerI = cms.ESProducer("ESTestProducerI")
process.esTestProducerJ = cms.ESProducer("ESTestProducerJ")
process.esTestProducerK = cms.ESProducer("ESTestProducerK")
process.esTestProducerK1 = cms.ESProducer("ESTestProducerK")
process.aPrefer = cms.ESPrefer("ESTestProducerK", "esTestProducerK")

process.esTestProducerH1 = cms.ESProducer("ESTestProducerH",
    appendToDataLabel = cms.string('abc')
)

process.esTestAnalyzerB = cms.EDAnalyzer("ESTestAnalyzerB",
    runsToGetDataFor = cms.vint32(1,2,3,4,5,6,7,8,9,10)
)

process.esTestAnalyzerK = cms.EDAnalyzer("ESTestAnalyzerK",
    runsToGetDataFor = cms.vint32(1,2,3,4,5,6,7,8,9,10)
)

process.path1 = cms.Path(process.esTestAnalyzerB*process.esTestAnalyzerK)

# ---------------------------------------------------------
# Test the sharing of ESTestProducerB as things associated
# with dependent records change.

# In this first case, nothing is different and sharing should
# occur. Note this is visible in the output because DataB
# is not gotten in event 2 and that counter in the producer
# is only incremented after a get call causes the produce
# function to be called.

process1 = cms.Process("TEST1")
process.subProcess = cms.SubProcess(process1)

process1.emptyESSourceB = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordB"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process1.emptyESSourceC = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordC"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process1.emptyESSourceD = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordD"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process1.emptyESSourceE = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordE"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process1.emptyESSourceF = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordF"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process1.emptyESSourceG = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordG"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process1.emptyESSourceH = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordH"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process1.emptyESSourceI = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordI"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process1.emptyESSourceJ = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordJ"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process1.emptyESSourceK = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordK"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process1.esTestProducerB = cms.ESProducer("ESTestProducerB")
process1.esTestProducerC = cms.ESProducer("ESTestProducerC")
process1.esTestProducerD = cms.ESProducer("ESTestProducerD")
process1.esTestProducerE = cms.ESProducer("ESTestProducerE")
process1.esTestProducerF = cms.ESProducer("ESTestProducerF")
process1.esTestProducerG = cms.ESProducer("ESTestProducerG")
process1.esTestProducerH = cms.ESProducer("ESTestProducerH")
process1.esTestProducerI = cms.ESProducer("ESTestProducerI")
process1.esTestProducerJ = cms.ESProducer("ESTestProducerJ")
process1.esTestProducerK = cms.ESProducer("ESTestProducerK")
process1.esTestProducerK1 = cms.ESProducer("ESTestProducerK")
process1.aPrefer = cms.ESPrefer("ESTestProducerK", "esTestProducerK")

process1.esTestProducerH1 = cms.ESProducer("ESTestProducerH",
    appendToDataLabel = cms.string('abc')
)

process1.esTestAnalyzerB = cms.EDAnalyzer("ESTestAnalyzerB",
    runsToGetDataFor = cms.vint32(1,3,4,5,6,7,8,9,10)
)

process1.esTestAnalyzerK = cms.EDAnalyzer("ESTestAnalyzerK",
    runsToGetDataFor = cms.vint32(1,3,4,5,6,7,8,9,10)
)

process1.path1 = cms.Path(process1.esTestAnalyzerB*process1.esTestAnalyzerK)

# ---------------------------------------------------------

# Change the value of a tracked parameter in an ESProducer
# associated with record H ( the module is labeled
# esTestProducerH1). Do not get data for event 3.
# Shows the esTestProducerB is not shared.

process2 = cms.Process("TEST2")
process1.subProcess = cms.SubProcess(process2)

process2.emptyESSourceB = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordB"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process2.emptyESSourceC = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordC"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process2.emptyESSourceD = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordD"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process2.emptyESSourceE = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordE"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process2.emptyESSourceF = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordF"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process2.emptyESSourceG = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordG"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process2.emptyESSourceH = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordH"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process2.emptyESSourceI = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordI"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process2.emptyESSourceJ = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordJ"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process2.emptyESSourceK = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordK"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process2.esTestProducerB = cms.ESProducer("ESTestProducerB")
process2.esTestProducerC = cms.ESProducer("ESTestProducerC")
process2.esTestProducerD = cms.ESProducer("ESTestProducerD")
process2.esTestProducerE = cms.ESProducer("ESTestProducerE")
process2.esTestProducerF = cms.ESProducer("ESTestProducerF")
process2.esTestProducerG = cms.ESProducer("ESTestProducerG")
process2.esTestProducerH = cms.ESProducer("ESTestProducerH")
process2.esTestProducerI = cms.ESProducer("ESTestProducerI")
process2.esTestProducerJ = cms.ESProducer("ESTestProducerJ")
process2.esTestProducerK = cms.ESProducer("ESTestProducerK",
    x = cms.untracked.string("x")
)

process2.esTestProducerH1 = cms.ESProducer("ESTestProducerH",
    appendToDataLabel = cms.string('abcd')
)

process2.esTestAnalyzerB = cms.EDAnalyzer("ESTestAnalyzerB",
    runsToGetDataFor = cms.vint32(1,2,4,5,6,7,8,9,10)
)

process2.esTestAnalyzerK = cms.EDAnalyzer("ESTestAnalyzerK",
    runsToGetDataFor = cms.vint32(1,4,5,6,7,8,9,10)
)

process2.path1 = cms.Path(process2.esTestAnalyzerB*process2.esTestAnalyzerK)

# ---------------------------------------------------------

# Change the value of a untracked parameter in an ESProducer
# associated with record H ( the module is labeled
# esTestProducerH1). Do not get data for event 4.
# Shows the esTestProducerB is not shared.

process3 = cms.Process("TEST3")
process2.subProcess = cms.SubProcess(process3)

process3.emptyESSourceB = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordB"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process3.emptyESSourceC = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordC"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process3.emptyESSourceD = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordD"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process3.emptyESSourceE = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordE"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process3.emptyESSourceF = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordF"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process3.emptyESSourceG = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordG"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process3.emptyESSourceH = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordH"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process3.esTestProducerB = cms.ESProducer("ESTestProducerB")
process3.esTestProducerC = cms.ESProducer("ESTestProducerC")
process3.esTestProducerD = cms.ESProducer("ESTestProducerD")
process3.esTestProducerE = cms.ESProducer("ESTestProducerE")
process3.esTestProducerF = cms.ESProducer("ESTestProducerF")
process3.esTestProducerG = cms.ESProducer("ESTestProducerG")
process3.esTestProducerH = cms.ESProducer("ESTestProducerH")

process3.esTestProducerH1 = cms.ESProducer("ESTestProducerH",
    appendToDataLabel = cms.string('abc'),
    test = cms.untracked.string('xyz')
)

process3.esTestAnalyzerB = cms.EDAnalyzer("ESTestAnalyzerB",
    runsToGetDataFor = cms.vint32(1,2,3,5,6,7,8,9,10)
)

process3.path1 = cms.Path(process3.esTestAnalyzerB)

# ---------------------------------------------------------

# This one should share with process3, not process1
# Do not get data for events 4 and 5.


process4 = cms.Process("TEST4")
process3.subProcess = cms.SubProcess(process4)

process4.emptyESSourceB = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordB"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process4.emptyESSourceC = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordC"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process4.emptyESSourceD = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordD"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process4.emptyESSourceE = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordE"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process4.emptyESSourceF = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordF"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process4.emptyESSourceG = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordG"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process4.emptyESSourceH = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordH"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process4.esTestProducerB = cms.ESProducer("ESTestProducerB")
process4.esTestProducerC = cms.ESProducer("ESTestProducerC")
process4.esTestProducerD = cms.ESProducer("ESTestProducerD")
process4.esTestProducerE = cms.ESProducer("ESTestProducerE")
process4.esTestProducerF = cms.ESProducer("ESTestProducerF")
process4.esTestProducerG = cms.ESProducer("ESTestProducerG")
process4.esTestProducerH = cms.ESProducer("ESTestProducerH")

process4.esTestProducerH1 = cms.ESProducer("ESTestProducerH",
    appendToDataLabel = cms.string('abc'),
    test = cms.untracked.string('xyz')
)

process4.esTestAnalyzerB = cms.EDAnalyzer("ESTestAnalyzerB",
    runsToGetDataFor = cms.vint32(1,2,3,6,7,8,9,10)
)

process4.path1 = cms.Path(process4.esTestAnalyzerB)

# ---------------------------------------------------------

# This one is like the top level process except esTestProducerH1
# has been removed completely. Does not do the get
# for events 1 to 3.


process5 = cms.Process("TEST5")
process4.subProcess = cms.SubProcess(process5)

process5.emptyESSourceB = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordB"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process5.emptyESSourceC = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordC"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process5.emptyESSourceD = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordD"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process5.emptyESSourceE = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordE"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process5.emptyESSourceF = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordF"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process5.emptyESSourceG = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordG"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process5.emptyESSourceH = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordH"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process5.esTestProducerB = cms.ESProducer("ESTestProducerB")
process5.esTestProducerC = cms.ESProducer("ESTestProducerC")
process5.esTestProducerD = cms.ESProducer("ESTestProducerD")
process5.esTestProducerE = cms.ESProducer("ESTestProducerE")
process5.esTestProducerF = cms.ESProducer("ESTestProducerF")
process5.esTestProducerG = cms.ESProducer("ESTestProducerG")
process5.esTestProducerH = cms.ESProducer("ESTestProducerH")

process5.esTestAnalyzerB = cms.EDAnalyzer("ESTestAnalyzerB",
    runsToGetDataFor = cms.vint32(4,5,6,7,8,9,10)
)

process5.path1 = cms.Path(process5.esTestAnalyzerB)

# ---------------------------------------------------------

# This one is like the top level process except esTestProducerH2
# has been added. Does not do the get for events 1 to 4.


process6 = cms.Process("TEST6")
process5.subProcess = cms.SubProcess(process6)

process6.emptyESSourceB = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordB"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process6.emptyESSourceC = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordC"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process6.emptyESSourceD = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordD"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process6.emptyESSourceE = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordE"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process6.emptyESSourceF = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordF"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process6.emptyESSourceG = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordG"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process6.emptyESSourceH = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordH"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process6.esTestProducerB = cms.ESProducer("ESTestProducerB")
process6.esTestProducerC = cms.ESProducer("ESTestProducerC")
process6.esTestProducerD = cms.ESProducer("ESTestProducerD")
process6.esTestProducerE = cms.ESProducer("ESTestProducerE")
process6.esTestProducerF = cms.ESProducer("ESTestProducerF")
process6.esTestProducerG = cms.ESProducer("ESTestProducerG")
process6.esTestProducerH = cms.ESProducer("ESTestProducerH")

process6.esTestProducerH1 = cms.ESProducer("ESTestProducerH",
    appendToDataLabel = cms.string('abc')
)

process6.esTestProducerH2 = cms.ESProducer("ESTestProducerH",
    appendToDataLabel = cms.string('xyz')
)

process6.esTestAnalyzerB = cms.EDAnalyzer("ESTestAnalyzerB",
    runsToGetDataFor = cms.vint32(5,6,7,8,9,10)
)

process6.path1 = cms.Path(process6.esTestAnalyzerB)

# ---------------------------------------------------------

# This one is like the top level process except an extra
# ESSource has been added for record F. Does not do the
# get for events 1 to 5.


process7 = cms.Process("TEST7")
process6.subProcess = cms.SubProcess(process7)

process7.emptyESSourceB = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordB"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process7.emptyESSourceC = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordC"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process7.emptyESSourceD = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordD"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process7.emptyESSourceE = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordE"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process7.emptyESSourceF = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordF"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process7.emptyESSourceF1 = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordF"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process7.emptyESSourceG = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordG"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process7.emptyESSourceH = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordH"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process7.esTestProducerB = cms.ESProducer("ESTestProducerB")
process7.esTestProducerC = cms.ESProducer("ESTestProducerC")
process7.esTestProducerD = cms.ESProducer("ESTestProducerD")
process7.esTestProducerE = cms.ESProducer("ESTestProducerE")
process7.esTestProducerF = cms.ESProducer("ESTestProducerF")
process7.esTestProducerG = cms.ESProducer("ESTestProducerG")
process7.esTestProducerH = cms.ESProducer("ESTestProducerH")

process7.esTestProducerH1 = cms.ESProducer("ESTestProducerH",
    appendToDataLabel = cms.string('abc')
)

process7.esTestAnalyzerB = cms.EDAnalyzer("ESTestAnalyzerB",
    runsToGetDataFor = cms.vint32(6,7,8,9,10)
)

process7.path1 = cms.Path(process7.esTestAnalyzerB)

# ---------------------------------------------------------

# This one is like the top level process except the
# ESSource was removed for record D. Does not do the
# get for events 1 to 6.

process8 = cms.Process("TEST8")
process7.subProcess = cms.SubProcess(process8)

process8.emptyESSourceB = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordB"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process8.emptyESSourceC = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordC"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process8.emptyESSourceE = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordE"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process8.emptyESSourceF = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordF"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process8.emptyESSourceG = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordG"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process8.emptyESSourceH = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordH"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process8.esTestProducerB = cms.ESProducer("ESTestProducerB")
process8.esTestProducerC = cms.ESProducer("ESTestProducerC")
process8.esTestProducerD = cms.ESProducer("ESTestProducerD")
process8.esTestProducerE = cms.ESProducer("ESTestProducerE")
process8.esTestProducerF = cms.ESProducer("ESTestProducerF")
process8.esTestProducerG = cms.ESProducer("ESTestProducerG")
process8.esTestProducerH = cms.ESProducer("ESTestProducerH")

process8.esTestProducerH1 = cms.ESProducer("ESTestProducerH",
    appendToDataLabel = cms.string('abc')
)

process8.esTestAnalyzerB = cms.EDAnalyzer("ESTestAnalyzerB",
    runsToGetDataFor = cms.vint32(7,8,9,10)
)

process8.path1 = cms.Path(process8.esTestAnalyzerB)

# ---------------------------------------------------------

# This one is like the top level process except the
# configuration was modified for theESSource for
# record G. Does not do the get for events 1 to 7.

process9 = cms.Process("TEST9")
process8.subProcess = cms.SubProcess(process9)

process9.emptyESSourceB = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordB"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process9.emptyESSourceC = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordC"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process9.emptyESSourceD = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordD"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process9.emptyESSourceE = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordE"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process9.emptyESSourceF = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordF"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process9.emptyESSourceG = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordG"),
    firstValid = cms.vuint32(1),
    iovIsRunNotTime = cms.bool(True)
)

process9.emptyESSourceH = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordH"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process9.esTestProducerB = cms.ESProducer("ESTestProducerB")
process9.esTestProducerC = cms.ESProducer("ESTestProducerC")
process9.esTestProducerD = cms.ESProducer("ESTestProducerD")
process9.esTestProducerE = cms.ESProducer("ESTestProducerE")
process9.esTestProducerF = cms.ESProducer("ESTestProducerF")
process9.esTestProducerG = cms.ESProducer("ESTestProducerG")
process9.esTestProducerH = cms.ESProducer("ESTestProducerH")

process9.esTestProducerH1 = cms.ESProducer("ESTestProducerH",
    appendToDataLabel = cms.string('abc')
)

process9.esTestAnalyzerB = cms.EDAnalyzer("ESTestAnalyzerB",
    runsToGetDataFor = cms.vint32(8,9,10)
)

process9.path1 = cms.Path(process9.esTestAnalyzerB)

# ---------------------------------------------------------

# Change the value of a tracked parameter in an ESProducer
# associated with record H ( the module is labeled
# esTestProducerH1). Do not get data for event 6.
# Shows the esTestProducerB is not shared.

process10 = cms.Process("TEST10")
process9.subProcess = cms.SubProcess(process10)

process10.emptyESSourceB = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordB"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process10.emptyESSourceC = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordC"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process10.emptyESSourceD = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordD"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process10.emptyESSourceE = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordE"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process10.emptyESSourceF = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordF"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process10.emptyESSourceG = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordG"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process10.emptyESSourceH = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordH"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process10.esTestProducerB = cms.ESProducer("ESTestProducerB")
process10.esTestProducerC = cms.ESProducer("ESTestProducerC")
process10.esTestProducerD = cms.ESProducer("ESTestProducerD")
process10.esTestProducerE = cms.ESProducer("ESTestProducerE")
process10.esTestProducerF = cms.ESProducer("ESTestProducerF")
process10.esTestProducerG = cms.ESProducer("ESTestProducerG")
process10.esTestProducerH = cms.ESProducer("ESTestProducerH")

process10.esTestProducerH1 = cms.ESProducer("ESTestProducerH",
    appendToDataLabel = cms.string('abc'),
    x = cms.string('abc')                                            
)

process10.esTestAnalyzerB = cms.EDAnalyzer("ESTestAnalyzerB",
    runsToGetDataFor = cms.vint32(1,2,3,4,5,7,8,9,10)
)

process10.path1 = cms.Path(process10.esTestAnalyzerB)

