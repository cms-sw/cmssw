import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.INFO.limit = 100

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

process.emptyESSourceA = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process.emptyESSourceA1 = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1),
    iovIsRunNotTime = cms.bool(True)
)

process.emptyESSourceZ = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordZ"),
    firstValid = cms.vuint32(1,3,5,7,9),
    iovIsRunNotTime = cms.bool(True)
)

process.esTestProducerA = cms.ESProducer("ESTestProducerA")

process.esTestProducerA1 = cms.ESProducer("ESTestProducerA",
    appendToDataLabel = cms.string('abc')
)

process.esTestProducerAZ = cms.ESProducer("ESTestProducerAZ")

process.esTestAnalyzerA = cms.EDAnalyzer("ESTestAnalyzerA",
    runsToGetDataFor = cms.vint32(1,2,3,4,5,6,7,8,9,10)
)

process.esTestAnalyzerAZ = cms.EDAnalyzer("ESTestAnalyzerAZ",
    runsToGetDataFor = cms.vint32(1,2,3,4,5,6,7,8,9,10)
)

process.path1 = cms.Path(process.esTestAnalyzerA*process.esTestAnalyzerAZ)

# -----------------------------------------------------------
# The primary goal is to test the sharing of ESProducers
# between SubProcess's and the top level process. We do this
# in a series of many SubProcess's

# The first SubProcess includes an ESTestProducerA
# which should be shared between this SubProcess
# and the previous process.  This is visible in the output
# because the data value printed by the ESTestAnalyzerA is
# incremented every time the ESTestProducer::produce method
# is called. This produce method will be called if something
# gets the data it produces while the event is being
# processed. In the first process the data is gotten
# on the first 9 events so there and for anything that
# shares its producer the values will increment
# on those events. In this SubProcess the data is not
# gotten on event 2 so the counter will not be incremented
# on that event and so a different value will get printed
# if this process is not sharing the ESProducer from the
# previous process.

process1 = cms.Process("TEST1")
process.subProcess = cms.SubProcess(process1)

process1.emptyESSourceA = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process1.emptyESSourceA1 = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1),
    iovIsRunNotTime = cms.bool(True)
)

process1.emptyESSourceZ = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordZ"),
    firstValid = cms.vuint32(1,3,5,7,9),
    iovIsRunNotTime = cms.bool(True)
)

process1.esTestProducerA = cms.ESProducer("ESTestProducerA")

process1.esTestProducerA1 = cms.ESProducer("ESTestProducerA",
    appendToDataLabel = cms.string('abc')
)

process1.esTestProducerAZ = cms.ESProducer("ESTestProducerAZ")

process1.esTestAnalyzerA = cms.EDAnalyzer("ESTestAnalyzerA",
    runsToGetDataFor = cms.vint32(1,3,4,5,6,7,8,9,10)
)

process1.path1 = cms.Path(process1.esTestAnalyzerA)


# ---------------------------------------------------------------
# The ESTestProducer should not shared in this SubProcess
# because of a difference in a tracked parameter

process2 = cms.Process("TEST2")
process1.subProcess = cms.SubProcess(process2)

process2.emptyESSourceA = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process2.emptyESSourceA1 = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1),
    iovIsRunNotTime = cms.bool(True)
)

process2.emptyESSourceZ = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordZ"),
    firstValid = cms.vuint32(1,3,5,7,9),
    iovIsRunNotTime = cms.bool(True)
)

process2.esTestProducerA = cms.ESProducer("ESTestProducerA",
    dummy = cms.string("test")
)

process2.esTestProducerA1 = cms.ESProducer("ESTestProducerA",
    appendToDataLabel = cms.string('abc')
)

process2.esTestProducerAZ = cms.ESProducer("ESTestProducerAZ")

process2.esTestAnalyzerA = cms.EDAnalyzer("ESTestAnalyzerA",
    runsToGetDataFor = cms.vint32(1,3,4,5,6,7,8,9,10)
)

process2.path1 = cms.Path(process2.esTestAnalyzerA)

# ---------------------------------------------------------------
# The ESTestProducer should not shared in this SubProcess
# because of a difference in an untracked parameter
# Data is not gotten on event 3.

process3 = cms.Process("TEST3")
process2.subProcess = cms.SubProcess(process3)

process3.emptyESSourceA = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process3.emptyESSourceA1 = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1),
    iovIsRunNotTime = cms.bool(True)
)

process3.emptyESSourceZ = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordZ"),
    firstValid = cms.vuint32(1,3,5,7,9),
    iovIsRunNotTime = cms.bool(True)
)

process3.esTestProducerA = cms.ESProducer("ESTestProducerA",
    dummy = cms.untracked.string("test")
)

process3.esTestProducerA1 = cms.ESProducer("ESTestProducerA",
    appendToDataLabel = cms.string('abc')
)

process3.esTestProducerAZ = cms.ESProducer("ESTestProducerAZ")

process3.esTestAnalyzerA = cms.EDAnalyzer("ESTestAnalyzerA",
    runsToGetDataFor = cms.vint32(1,2,4,5,6,7,8,9,10)
)

process3.path1 = cms.Path(process3.esTestAnalyzerA)

# ---------------------------------------------------------------
# The ESTestProducer should not shared in this SubProcess
# because of an extra ESProducer adding data to the same record
# even though it has one ESProducer whose configuration matches
# exactly. No data gotten on event 4 and 5.

process4 = cms.Process("TEST4")
process3.subProcess = cms.SubProcess(process4)

process4.emptyESSourceA = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process4.emptyESSourceA1 = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1),
    iovIsRunNotTime = cms.bool(True)
)

process4.emptyESSourceZ = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordZ"),
    firstValid = cms.vuint32(1,3,5,7,9),
    iovIsRunNotTime = cms.bool(True)
)

process4.esTestProducerA = cms.ESProducer("ESTestProducerA")

process4.esTestProducerA1 = cms.ESProducer("ESTestProducerA",
    appendToDataLabel = cms.string('abc')
)

process4.esTestProducerA2 = cms.ESProducer("ESTestProducerA",
    appendToDataLabel = cms.string('abcd')
)

process4.esTestProducerAZ = cms.ESProducer("ESTestProducerAZ")

process4.esTestAnalyzerA = cms.EDAnalyzer("ESTestAnalyzerA",
    runsToGetDataFor = cms.vint32(1,2,3,6,7,8,9,10)
)

process4.esTestAnalyzerAZ = cms.EDAnalyzer("ESTestAnalyzerAZ",
    runsToGetDataFor = cms.vint32(3,4,5,6,7,8,9,10)
)

process4.path1 = cms.Path(process4.esTestAnalyzerA*process4.esTestAnalyzerAZ)

# ---------------------------------------------------------------
# Same as previous. Should share with the previous one
# not anything earlier. No data gotten on events 4, 5, and 6.

process5 = cms.Process("TEST5")
process4.subProcess = cms.SubProcess(process5)

process5.emptyESSourceA = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process5.emptyESSourceA1 = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1),
    iovIsRunNotTime = cms.bool(True)
)

process5.emptyESSourceZ = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordZ"),
    firstValid = cms.vuint32(1,3,5,7,9),
    iovIsRunNotTime = cms.bool(True)
)

process5.esTestProducerA = cms.ESProducer("ESTestProducerA",
)

process5.esTestProducerA1 = cms.ESProducer("ESTestProducerA",
    appendToDataLabel = cms.string('abc')
)

process5.esTestProducerA2 = cms.ESProducer("ESTestProducerA",
    appendToDataLabel = cms.string('abcd')
)

process5.esTestProducerAZ = cms.ESProducer("ESTestProducerAZ")

process5.esTestAnalyzerA = cms.EDAnalyzer("ESTestAnalyzerA",
    runsToGetDataFor = cms.vint32(1,2,3,7,8,9,10)
)

process5.esTestAnalyzerAZ = cms.EDAnalyzer("ESTestAnalyzerAZ",
    runsToGetDataFor = cms.vint32(5,6,7,8,9,10)
)

process5.path1 = cms.Path(process5.esTestAnalyzerA*process5.esTestAnalyzerAZ)

# ---------------------------------------------------------------
# Same as original except one less ESProducer for record.
# No data gotten on event 7

process6 = cms.Process("TEST6")
process5.subProcess = cms.SubProcess(process6)

process6.emptyESSourceA = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process6.emptyESSourceA1 = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1),
    iovIsRunNotTime = cms.bool(True)
)

process6.emptyESSourceZ = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordZ"),
    firstValid = cms.vuint32(1,3,5,7,9),
    iovIsRunNotTime = cms.bool(True)
)

process6.esTestProducerA = cms.ESProducer("ESTestProducerA")

process6.esTestProducerAZ = cms.ESProducer("ESTestProducerAZ")

process6.esTestAnalyzerA = cms.EDAnalyzer("ESTestAnalyzerA",
    runsToGetDataFor = cms.vint32(1,2,3,4,5,6,8,9,10)
)

process6.path1 = cms.Path(process6.esTestAnalyzerA)

# ---------------------------------------------------------------
# Same as original except an ESProducer associated with
# same record has a differing ParameterSet. No data
# gotten on event 8.

process7 = cms.Process("TEST7")
process6.subProcess = cms.SubProcess(process7)

process7.emptyESSourceA = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process7.emptyESSourceA1 = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1),
    iovIsRunNotTime = cms.bool(True)
)

process7.emptyESSourceZ = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordZ"),
    firstValid = cms.vuint32(1,3,5,7,9),
    iovIsRunNotTime = cms.bool(True)
)

process7.esTestProducerA = cms.ESProducer("ESTestProducerA")

process7.esTestProducerA1 = cms.ESProducer("ESTestProducerA",
    appendToDataLabel = cms.string('xyz')
)

process7.esTestProducerAZ = cms.ESProducer("ESTestProducerAZ")

process7.esTestAnalyzerA = cms.EDAnalyzer("ESTestAnalyzerA",
    runsToGetDataFor = cms.vint32(1,2,3,4,5,6,7,9,10)
)

process7.path1 = cms.Path(process7.esTestAnalyzerA)

# ---------------------------------------------------------------
# Same as original except an ESProducer associated with
# same record has a differing ParameterSet, an untracked
# difference. No data gotten on event 1 to 4.

process8 = cms.Process("TEST8")
process7.subProcess = cms.SubProcess(process8)

process8.emptyESSourceA = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process8.emptyESSourceA1 = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1),
    iovIsRunNotTime = cms.bool(True)
)

process8.emptyESSourceZ = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordZ"),
    firstValid = cms.vuint32(1,3,5,7,9),
    iovIsRunNotTime = cms.bool(True)
)

process8.esTestProducerA = cms.ESProducer("ESTestProducerA")

process8.esTestProducerA1 = cms.ESProducer("ESTestProducerA",
    appendToDataLabel = cms.string('abc'),
    test = cms.untracked.string('q')
)

process8.esTestProducerAZ = cms.ESProducer("ESTestProducerAZ")


process8.esTestAnalyzerA = cms.EDAnalyzer("ESTestAnalyzerA",
    runsToGetDataFor = cms.vint32(5,6,7,8,9,10)
)

process8.path1 = cms.Path(process8.esTestAnalyzerA)

# ---------------------------------------------------------------
# Same as original except one less ESSource associated with the
# same record. No data gotten on event 1 to 5.

process9 = cms.Process("TEST9")
process8.subProcess = cms.SubProcess(process9)

process9.emptyESSourceA = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process9.emptyESSourceZ = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordZ"),
    firstValid = cms.vuint32(1,3,5,7,9),
    iovIsRunNotTime = cms.bool(True)
)

process9.esTestProducerA = cms.ESProducer("ESTestProducerA")

process9.esTestProducerA1 = cms.ESProducer("ESTestProducerA",
    appendToDataLabel = cms.string('abc')
)

process9.esTestProducerAZ = cms.ESProducer("ESTestProducerAZ")

process9.esTestAnalyzerA = cms.EDAnalyzer("ESTestAnalyzerA",
    runsToGetDataFor = cms.vint32(6,7,8,9,10)
)

process9.path1 = cms.Path(process9.esTestAnalyzerA)

# ---------------------------------------------------------------
# Same as original except one more ESSource associated with the
# same record. No data gotten on event 1 to 6.

process10 = cms.Process("TEST10")
process9.subProcess = cms.SubProcess(process10)

process10.emptyESSourceA = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process10.emptyESSourceA1 = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1),
    iovIsRunNotTime = cms.bool(True)
)

process10.emptyESSourceA2 = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1),
    iovIsRunNotTime = cms.bool(True)
)

process10.emptyESSourceZ = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordZ"),
    firstValid = cms.vuint32(1,3,5,7,9),
    iovIsRunNotTime = cms.bool(True)
)

process10.esTestProducerA = cms.ESProducer("ESTestProducerA")

process10.esTestProducerA1 = cms.ESProducer("ESTestProducerA",
    appendToDataLabel = cms.string('abc')
)

process10.esTestProducerAZ = cms.ESProducer("ESTestProducerAZ")

process10.esTestAnalyzerA = cms.EDAnalyzer("ESTestAnalyzerA",
    runsToGetDataFor = cms.vint32(7,8,9,10)
)

process10.path1 = cms.Path(process10.esTestAnalyzerA)

# ---------------------------------------------------------------
# Same as original except an ESSource associated with the
# same record has a differing configuration. No data gotten on event 1 to 7.

process11 = cms.Process("TEST11")
process10.subProcess = cms.SubProcess(process11)

process11.emptyESSourceA = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process11.emptyESSourceA1 = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1,2),
    iovIsRunNotTime = cms.bool(True)
)

process11.emptyESSourceZ = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordZ"),
    firstValid = cms.vuint32(1,3,5,7,9),
    iovIsRunNotTime = cms.bool(True)
)

process11.esTestProducerA = cms.ESProducer("ESTestProducerA")

process11.esTestProducerA1 = cms.ESProducer("ESTestProducerA",
    appendToDataLabel = cms.string('abc')
)

process11.esTestProducerAZ = cms.ESProducer("ESTestProducerAZ")

process11.esTestAnalyzerA = cms.EDAnalyzer("ESTestAnalyzerA",
    runsToGetDataFor = cms.vint32(8,9,10)
)

process11.path1 = cms.Path(process11.esTestAnalyzerA)

# ---------------------------------------------------------------
# Same as process 2 and their ESProducers should be shared.
# No data gotten on event 1 to 8. Except for ESTestProducerAZ
# because the emptyESSourceZ has a differing configuration.

process12 = cms.Process("TEST12")
process11.subProcess = cms.SubProcess(process12)

process12.emptyESSourceA = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1,2,3,4,5,6,7,8,9),
    iovIsRunNotTime = cms.bool(True)
)

process12.emptyESSourceA1 = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1),
    iovIsRunNotTime = cms.bool(True)
)

process12.emptyESSourceZ = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordZ"),
    firstValid = cms.vuint32(1,2,3,5,7,9),
    iovIsRunNotTime = cms.bool(True)
)

process12.esTestProducerA = cms.ESProducer("ESTestProducerA",
    dummy = cms.string("test")
)

process12.esTestProducerA1 = cms.ESProducer("ESTestProducerA",
    appendToDataLabel = cms.string('abc')
)

process12.esTestProducerAZ = cms.ESProducer("ESTestProducerAZ")

process12.esTestAnalyzerA = cms.EDAnalyzer("ESTestAnalyzerA",
    runsToGetDataFor = cms.vint32(9,10)
)

process12.path1 = cms.Path(process12.esTestAnalyzerA)
