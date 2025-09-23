import FWCore.ParameterSet.Config as cms

process = cms.Process("FALLBACK3")
from FWIO.RNTupleTempInput.modules import RNTupleTempSource
process.source = RNTupleTempSource(
    fileNames = cms.untracked.vstring(
        'file:testNoProcessFallback2.root',
    )
)

from FWCore.Modules.modules import timestudy_SleepingProducer
process.intProducer = timestudy_SleepingProducer( ivalue = 3, eventTimes = [0] )

from FWCore.Modules.modules import EventIDFilter
process.keeper = EventIDFilter(
        eventsToPass = [cms.EventID(1,1,5), cms.EventID(1,1,6), cms.EventID(1,1,7), cms.EventID(1,1,8)]
)

# The matrix of passed Events for the three processes (X is pass, - is fail)
# EventID      FALLBACK1   FALLBACK2   FALLBACK3
# (1,1,1)         -           -           -
# (1,1,2)         X           -           -
# (1,1,3)         -           X           -
# (1,1,4)         X           X           -
# (1,1,5)         -           -           X
# (1,1,6)         X           -           X
# (1,1,7)         -           X           X
# (1,1,8)         X           X           X



process.p = cms.Path(process.keeper + process.intProducer )

process.testerNoProcessMissing = cms.EDAnalyzer("BuiltinIntTestAnalyzer",
    moduleLabel = cms.untracked.InputTag("intProducer"),
    valueMustBeMissing = cms.untracked.bool(True),
    valueMustMatch = cms.untracked.int32(1)  # Ignored if valueMustBeMissing is true
)
process.testerSkipCurrentProcessMissing = cms.EDAnalyzer("BuiltinIntTestAnalyzer",
    moduleLabel = cms.untracked.InputTag("intProducer", processName=cms.InputTag.skipCurrentProcess()),
    valueMustBeMissing = cms.untracked.bool(True),
    valueMustMatch = cms.untracked.int32(1)  # Ignored if valueMustBeMissing is true
)

process.testerCurrentProcessMissing = cms.EDAnalyzer("BuiltinIntTestAnalyzer",
    moduleLabel = cms.untracked.InputTag("intProducer", processName=cms.InputTag.currentProcess()),
    valueMustBeMissing = cms.untracked.bool(True),
    valueMustMatch = cms.untracked.int32(1)  # Ignored if valueMustBeMissing is true
)


process.keeper1 = EventIDFilter(
        eventsToPass = [cms.EventID(1,1,1)]
)
process.p1 = cms.Path(process.keeper1 + process.testerNoProcessMissing + process.testerSkipCurrentProcessMissing + process.testerCurrentProcessMissing)

process.testerNoProcess1 = cms.EDAnalyzer("BuiltinIntTestAnalyzer",
    moduleLabel = cms.untracked.InputTag("intProducer"),
    valueMustMatch = cms.untracked.int32(1),
    valueMustBeMissing = cms.untracked.bool(False)
)

process.testerSkipCurrentProcess1 = cms.EDAnalyzer("BuiltinIntTestAnalyzer",
    moduleLabel = cms.untracked.InputTag("intProducer", processName=cms.InputTag.skipCurrentProcess()),
    valueMustMatch = cms.untracked.int32(1),
    valueMustBeMissing = cms.untracked.bool(False)
)

process.keeper2 = EventIDFilter(
        eventsToPass = [cms.EventID(1,1,2)]
)
process.p2 = cms.Path(process.keeper2 + process.testerNoProcessMissing + process.testerSkipCurrentProcessMissing + process.testerCurrentProcessMissing)

process.keeper3 = EventIDFilter(
        eventsToPass = [cms.EventID(1,1,3)]
)

process.testerNoProcess2 = cms.EDAnalyzer("BuiltinIntTestAnalyzer",
    moduleLabel = cms.untracked.InputTag("intProducer"),
    valueMustMatch = cms.untracked.int32(2),
    valueMustBeMissing = cms.untracked.bool(False)
)

process.testerSkipCurrentProcess2 = cms.EDAnalyzer("BuiltinIntTestAnalyzer",
    moduleLabel = cms.untracked.InputTag("intProducer", processName=cms.InputTag.skipCurrentProcess()),
    valueMustMatch = cms.untracked.int32(2),
    valueMustBeMissing = cms.untracked.bool(False)
)

process.p3 = cms.Path(process.keeper3 + process.testerNoProcessMissing + process.testerSkipCurrentProcess2 + process.testerCurrentProcessMissing)

process.keeper4 = EventIDFilter(
        eventsToPass = [cms.EventID(1,1,4)]
)
process.p4 = cms.Path(process.keeper4 + process.testerNoProcessMissing + process.testerSkipCurrentProcess2 + process.testerCurrentProcessMissing)

process.testerNoProcess3 = cms.EDAnalyzer("BuiltinIntTestAnalyzer",
    moduleLabel = cms.untracked.InputTag("intProducer"),
    valueMustMatch = cms.untracked.int32(3),
    valueMustBeMissing = cms.untracked.bool(False)
)
process.testerCurrentProcess3 = cms.EDAnalyzer("BuiltinIntTestAnalyzer",
    moduleLabel = cms.untracked.InputTag("intProducer", processName=cms.InputTag.currentProcess()),
    valueMustBeMissing = cms.untracked.bool(False),
    valueMustMatch = cms.untracked.int32(3)
)


process.keeper5 = EventIDFilter(
        eventsToPass = [cms.EventID(1,1,5)]
)
process.p5 = cms.Path(process.keeper5 + process.testerNoProcess3 + process.testerSkipCurrentProcessMissing + process.testerCurrentProcess3)

process.keeper6 = EventIDFilter(
        eventsToPass = [cms.EventID(1,1,6)]
)
process.p6 = cms.Path(process.keeper6 + process.testerNoProcess3 + process.testerSkipCurrentProcessMissing + process.testerCurrentProcess3)

process.keeper7 = EventIDFilter(
        eventsToPass = [cms.EventID(1,1,7)]
)
process.p7 = cms.Path(process.keeper7 + process.testerNoProcess3 + process.testerSkipCurrentProcess2 + process.testerCurrentProcess3)

process.keeper8 = EventIDFilter(
        eventsToPass = [cms.EventID(1,1,8)]
)
process.p8 = cms.Path(process.keeper8 + process.testerNoProcess3 + process.testerSkipCurrentProcess2 + process.testerCurrentProcess3)

#from FWCore.Modules.modules import EventContentAnalyzer
#process.dump = EventContentAnalyzer()
#process.dumpPath = cms.Path(process.dump)