#test fallback logic when current process does not produce the product
import FWCore.ParameterSet.Config as cms

process = cms.Process("FALLBACK4")
from FWIO.RNTupleTempInput.modules import RNTupleTempSource
process.source = RNTupleTempSource(
    fileNames = cms.untracked.vstring(
        'file:testNoProcessFallback2.root',
    )
)

process.maxEvents.input = 4

# The matrix of passed Events for the three processes (X is pass, - is fail)
# EventID      FALLBACK1   FALLBACK2   FALLBACK4
# (1,1,1)         -           -           -
# (1,1,2)         X           -           -
# (1,1,3)         -           X           -
# (1,1,4)         X           X           -




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

from FWCore.Modules.modules import EventIDFilter
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

process.p3 = cms.Path(process.keeper3 + process.testerNoProcess2 + process.testerSkipCurrentProcess2 + process.testerCurrentProcessMissing)

process.keeper4 = EventIDFilter(
        eventsToPass = [cms.EventID(1,1,4)]
)
process.p4 = cms.Path(process.keeper4 + process.testerNoProcess2 + process.testerSkipCurrentProcess2 + process.testerCurrentProcessMissing)


#from FWCore.Modules.modules import EventContentAnalyzer
#process.dump = EventContentAnalyzer()
#process.dumpPath = cms.Path(process.dump)