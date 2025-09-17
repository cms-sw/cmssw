import FWCore.ParameterSet.Config as cms

process = cms.Process("FALLBACK2")
from IOPool.Input.modules import PoolSource
process.source = PoolSource(
    fileNames = cms.untracked.vstring(
        'file:testNoProcessFallback1.root',
    )
)

from FWCore.Modules.modules import timestudy_SleepingProducer
process.intProducer = timestudy_SleepingProducer( ivalue = 2, eventTimes = [0] )

from FWCore.Modules.modules import EventIDFilter
process.keeper = EventIDFilter(
        eventsToPass = [cms.EventID(1,1,3), cms.EventID(1,1,4), cms.EventID(1,1,7), cms.EventID(1,1,8)]
)

from IOPool.Output.modules import PoolOutputModule
process.out = PoolOutputModule(
    fileName = 'testNoProcessFallback2.root',
)

process.p = cms.Path(process.keeper + process.intProducer )
process.e = cms.EndPath(process.out)