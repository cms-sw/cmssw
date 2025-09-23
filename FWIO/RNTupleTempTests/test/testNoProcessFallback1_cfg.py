import FWCore.ParameterSet.Config as cms

process = cms.Process("FALLBACK1")
from FWCore.Modules.modules import EmptySource
process.source = EmptySource(
    numberEventsInRun = 8,
    numberEventsInLuminosityBlock = 8,
)

process.maxEvents.input = 8

from FWCore.Modules.modules import timestudy_SleepingProducer
process.intProducer = timestudy_SleepingProducer( ivalue = 1, eventTimes = [0] )

from FWCore.Modules.modules import EventIDFilter
process.keeper = EventIDFilter(
        eventsToPass = [cms.EventID(1,1,2), cms.EventID(1,1,4), cms.EventID(1,1,6), cms.EventID(1,1,8)]
)

from FWIO.RNTupleTempOutput.modules import RNTupleTempOutputModule
process.out = RNTupleTempOutputModule(
    fileName = 'testNoProcessFallback1.root',
)

process.p = cms.Path(process.keeper + process.intProducer )
process.e = cms.EndPath(process.out)